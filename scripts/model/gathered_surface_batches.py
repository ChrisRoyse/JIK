"""Gather repeated windows from one owned source tile; keep mixed-batch arithmetic."""
import torch

from model import mixed_surface_batches as reference
from model.native import NativeInput
from model.stream_surface import schedule
from senses.contract import digest
from text.constellation_corpus import need


def stamp(tensor):
    return (id(tensor), tensor._version, tensor.data_ptr(), tensor.shape,
            tensor.stride(), tensor.dtype, tensor.device, tensor.requires_grad)


class SourceTile:
    """Process-local owned buffers, never an unchecked persistent reading cache."""
    def __init__(self, examples, config, specs):
        reference.validate(examples, config, specs, 128)
        self.examples, self.config, self.specs = tuple(examples), config, tuple(specs)
        count = sum(len(e.raw) + 2 for e in examples)
        need(count <= config.max_native_elements, 'GATHERED_SURFACE_BYTE_MEMORY')
        self.closed = False
        self.sources = tuple((e, tuple(e.native.items())) for e in examples)
        self.source_stamps = tuple(tuple((x, stamp(x.values), stamp(x.padding), stamp(x.absent))
                                        for x in e.native.values()) for e in examples)
        starts, tokens = [], []
        for e in examples:
            starts.append(len(tokens)); tokens.extend([256, *e.raw, 257])
        self.surface = torch.tensor(tokens, dtype=torch.int64)
        self.starts = torch.tensor(starts, dtype=torch.int64)
        self.denominators = torch.tensor([len(e.raw) + 1 for e in examples], dtype=torch.float32)
        self.banks, self.addresses = {}, {}
        owned = [self.surface, self.starts, self.denominators]
        for spec in specs:
            groups = {}
            for i, e in enumerate(examples):
                x = e.native[spec.name]
                need(x.values.device.type == 'cpu', 'GATHERED_SURFACE_HOST_INPUT')
                groups.setdefault(x.values.shape[1], []).append(i)
            self.banks[spec.name], self.addresses[spec.name] = {}, {}
            for length, indices in groups.items():
                values = torch.stack([examples[i].native[spec.name].values[0] for i in indices])
                self.banks[spec.name][length] = values; owned.append(values)
                for local, i in enumerate(indices): self.addresses[spec.name][i] = (length, local)
        self.owned_stamps = tuple((x, stamp(x)) for x in owned)
        self.owned_bytes = sum(x.numel() * x.element_size() for x in owned)
        self.fresh()

    def fresh(self, atoms=None):
        need(not self.closed, 'GATHERED_SURFACE_CLOSED')
        selected = range(len(self.examples)) if atoms is None else atoms
        for i in selected:
            need(type(i) is int and 0 <= i < len(self.examples), 'GATHERED_SURFACE_SOURCE_INDEX')
            e, slots = self.sources[i]
            need(tuple(e.native) == tuple(n for n, _ in slots)
                 and all(e.native[n] is x for n, x in slots), 'GATHERED_SURFACE_REBOUND_SOURCE')
            need(all(stamp(x.values) == v and stamp(x.padding) == p and stamp(x.absent) == a
                     for x, v, p, a in self.source_stamps[i]), 'GATHERED_SURFACE_CHANGED_SOURCE')
        current = [self.surface, self.starts, self.denominators]
        current.extend(x for group in self.banks.values() for x in group.values())
        need(len(current) == len(self.owned_stamps)
             and all(stamp(x) == saved for x, (_, saved) in zip(current, self.owned_stamps)),
             'GATHERED_SURFACE_CHANGED_CACHE')

    def close(self):
        self.closed = True
        self.sources = self.source_stamps = self.owned_stamps = self.examples = ()
        self.banks.clear(); self.addresses.clear()
        self.surface = self.starts = self.denominators = None

    def __enter__(self):
        self.fresh(); return self

    def __exit__(self, *_): self.close()

    def assemble(self, records):
        need(not self.closed, 'GATHERED_SURFACE_CLOSED')
        need(records and len(records) <= 512
             and all(type(i) is int and 0 <= i < len(self.examples) for i, _ in records),
             'GATHERED_SURFACE_RECORDS')
        used_atoms = sorted({i for i, _ in records}); self.fresh(used_atoms)
        examples, config, specs = self.examples, self.config, self.specs
        need(reference.fits(records, examples, config, specs), 'GATHERED_SURFACE_PHYSICAL_MEMORY')
        b = len(records); width = max(len(item['prefix']) for _, item in records)
        atoms = torch.tensor([i for i, _ in records], dtype=torch.int64)
        offsets = torch.tensor([item['offset'] for _, item in records], dtype=torch.int64)
        lengths = torch.tensor([len(item['prefix']) for _, item in records], dtype=torch.int64)
        left = torch.tensor([item['scored'][0] - item['offset'] for _, item in records], dtype=torch.int64)
        column = torch.arange(width)[None, :]; present = column < lengths[:, None]
        indices = self.starts[atoms, None] + offsets[:, None] + column
        prefix = self.surface[torch.where(present, indices, 0)].masked_fill(~present, 0)
        labels = self.surface[torch.where(present, indices + 1, 0)]
        labels.masked_fill_(~present | (column < left[:, None]), -100)
        meta, ids = [], []
        for i, item in records:
            e = examples[i]
            record = dict(atom=i, occurrence=e.occurrence, uid=e.uid, offset=item['offset'],
                          scored=item['scored'], prefix_length=len(item['prefix']))
            meta.append(record); ids.append(digest(dict(source=next(iter(e.native.values())).rows, **record)))
        native = {}
        for spec in specs:
            groups = {}; row_lengths = []
            for row, (i, _) in enumerate(records):
                length, address = self.addresses[spec.name][i]; row_lengths.append(length)
                dest, source = groups.setdefault(length, ([], [])); dest.append(row); source.append(address)
            n = max(groups)
            if len(groups) == 1:
                values = self.banks[spec.name][n].index_select(0, torch.tensor(groups[n][1]))
            else:
                values = torch.zeros((b, n, spec.dims), dtype=torch.float32)
                for length, (dest, source) in groups.items():
                    part = self.banks[spec.name][length].index_select(0, torch.tensor(source))
                    values[:, :length].index_copy_(0, torch.tensor(dest), part)
            padding = torch.arange(n)[None, :] >= torch.tensor(row_lengths)[:, None]
            native[spec.name] = NativeInput(spec.identity, tuple(ids), values, padding, torch.zeros(b, dtype=torch.bool))
            native[spec.name].check(spec, config)
        self.fresh(used_atoms)
        return reference.MixedSurfaceBatch(prefix, labels, offsets, self.denominators[atoms],
                                           native, tuple(meta), len(examples))


def batches(examples, config, specs, *, rows=128):
    """Same physical scheduling as phase80, with tile-owned gather buffers."""
    reference.validate(examples, config, specs, rows)
    with SourceTile(examples, config, specs) as tile:
        pending = []; width = 0; lengths = {s.name: 0 for s in specs}
        for i, e in enumerate(examples):
            native_lengths = {s.name: e.native[s.name].values.shape[1] for s in specs}
            for item in schedule(e.raw, width=config.max_prefix):
                new_width = max(width, len(item['prefix']))
                new_lengths = {n: max(lengths[n], v) for n, v in native_lengths.items()}
                if pending and (len(pending) == rows or not reference.layout_fits(
                        len(pending) + 1, new_width, new_lengths, config, specs)):
                    yield tile.assemble(pending); pending = []
                    new_width = len(item['prefix']); new_lengths = native_lengths.copy()
                need(reference.layout_fits(len(pending) + 1, new_width, new_lengths, config, specs),
                     'GATHERED_SURFACE_SINGLE_ROW_MEMORY')
                pending.append((i, item)); width = new_width; lengths = new_lengths
        if pending: yield tile.assemble(pending)
