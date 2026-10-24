"""Reuse saved handles while the identical tensor and version remain live.

Values, aliases, autograd edges and mutation checks are unchanged. Weak references
prevent either the cache or the original-object identity check retaining a graph.
"""
import weakref

from model.saved_gradient_storage import SavedStorage
from text.constellation_corpus import need


class SharedSavedStorage(SavedStorage):
    def __init__(self, allowance, device):
        super().__init__(allowance, device)
        self.handles = weakref.WeakValueDictionary()
        self.reused_handles = self.created_handles = self.live_handles = self.peak_handles = 0

    def pack(self, tensor):
        saved = self.handles.get(id(tensor))
        if saved is not None and saved.original() is tensor and saved.version == tensor._version:
            backing = tensor.untyped_storage()
            key = (str(tensor.device), backing.data_ptr(), backing.nbytes())
            old = saved.tensor
            # .data replacement can change storage/layout without advancing _version.
            same = (key == saved.key and old.dtype == tensor.dtype and old.layout == tensor.layout
                    and old.shape == tensor.shape and old.stride() == tensor.stride()
                    and old.storage_offset() == tensor.storage_offset()
                    and old.is_conj() == tensor.is_conj() and old.is_neg() == tensor.is_neg())
            if same:
                need(self.current <= self.limit, 'GRADIENT_STORAGE_MEMORY')
                self.packs += 1; self.reused_handles += 1
                return saved
        saved = super().pack(tensor)
        saved.original = weakref.ref(tensor)
        self.handles[id(tensor)] = saved
        self.created_handles += 1; self.live_handles += 1
        self.peak_handles = max(self.peak_handles, self.live_handles)
        return saved

    def release(self, key):
        super().release(key)
        self.live_handles -= 1

    def report(self):
        return dict(super().report(), reused_handles=self.reused_handles,
                    created_handles=self.created_handles, live_handles=self.live_handles,
                    peak_live_handles=self.peak_handles, weak_cache_entries=len(self.handles))
