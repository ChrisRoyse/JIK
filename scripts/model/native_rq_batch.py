"""Batch independent student atoms by complete native-window layout.

This is a separately pinned student runtime, not an instrument implementation.
The quantizer latent and typed outputs have the same definitions as the scalar
path. No attention, normalization or source addressing crosses an atom boundary.
"""
import torch

from model.native_rq_interventions import InterventionNativeRQ
from text.constellation_corpus import need


class BatchedInterventionNativeRQ(InterventionNativeRQ):
    def groups(self, counts):
        need(isinstance(counts, (list, tuple)) and counts, 'RQ_BATCH_EMPTY')
        groups = {}; tokens = elements = attention = 0
        for i, row in enumerate(counts):
            need(set(row) == set(self.roster) and all(type(w) is int and w > 0 for w in row.values()),
                 'RQ_BATCH_WINDOW_COUNTS')
            key = tuple(row[n] for n in self.roster); size = sum(key)+len(key)
            tokens += size; elements += sum(row[n]*self.dimensions[n] for n in self.roster)
            attention += self.config['heads']*size*size
            groups.setdefault(key, []).append(i)
        need(tokens <= self.config['max_native_tokens'] and elements <= self.config['max_native_elements']
             and attention <= self.config['max_attention_elements'], 'RQ_BATCH_PHYSICAL_MEMORY')
        return [(key, groups[key]) for key in sorted(groups)]

    def latent(self, frames):
        need(isinstance(frames, (list, tuple)) and frames, 'RQ_BATCH_EMPTY')
        for frame in frames:
            self.validate(frame)
            need(all(not x.requires_grad for x in frame.values()), 'RQ_BATCH_NATIVE_REQUIRES_GRAD')
        groups = self.groups([{n:len(f[n]) for n in self.roster} for f in frames])
        encoder = self.encoder; device = encoder.query.device
        atom_position = encoder.position(torch.zeros(1, dtype=torch.int64, device=device))[0]
        result = [None]*len(frames)
        for counts, indices in groups:
            tokens = []; queries = []; cursor = 0
            for lane, (name, windows) in enumerate(zip(self.roster, counts)):
                native = torch.stack([frames[i][name] for i in indices])
                normalized = (native-getattr(self, 'center_'+str(lane)))/getattr(self, 'variance_'+str(lane)).sqrt()
                local = encoder.position(torch.arange(windows, device=device))
                tokens.append(self.projectors[name](normalized)+encoder.kind[lane]+atom_position+local)
                cursor += windows; queries.append(cursor)
                query = encoder.query[lane]+encoder.kind[lane]+atom_position
                tokens.append(query[None, None].expand(len(indices), 1, -1)); cursor += 1
            x = torch.cat(tokens, 1)
            # Every token in this row belongs to one atom. All such positions
            # may attend to each other; independent batch rows cannot interact.
            mask = torch.zeros((cursor, cursor), dtype=torch.bool, device=device)
            for block in encoder.blocks: x = block(x, mask)
            selected = torch.tensor(queries, dtype=torch.int64, device=device)
            hidden = encoder.norm(x)[:, selected]
            values = self.to_latent(hidden.reshape(len(indices), -1))
            for i, value in zip(indices, values): result[i] = value
        return torch.stack(result)

    def reconstruct(self, latent, counts):
        need(latent.ndim == 2 and latent.shape == (len(counts), self.config['width'])
             and latent.dtype == torch.float32 and latent.device == self.encoder.query.device
             and bool(torch.isfinite(latent).all()), 'RQ_BATCH_LATENT')
        groups = self.groups(counts); result = [None]*len(counts)
        for sizes, indices in groups:
            z = latent[indices]; rows = [{} for _ in indices]
            for lane, (name, windows) in enumerate(zip(self.roster, sizes)):
                positions = self.encoder.position(torch.arange(windows, device=z.device))
                values = torch.cat((z[:, None].expand(-1, windows, -1),
                                    positions[None].expand(len(indices), -1, -1)), -1)
                residual = self.decoders[name](values)
                decoded = self.baseline(lane, windows)+getattr(self, 'variance_'+str(lane)).sqrt()*residual
                for row, native in zip(rows, decoded): row[name] = native
            for i, row in zip(indices, rows): result[i] = row
        return tuple(result)
