"""Bounded feedback caches for larger datasets; objectives and views are unchanged."""
from collections import OrderedDict
import torch
from utils.sfra_resident_feedback import ResidentWitnessFeedback


class BoundedWitnessFeedback(ResidentWitnessFeedback):
    def __init__(self, bank, forward_batch_size=128, cache_gib=4., reserve_gib=4.,
                 cpu_cache_gib=4., image_cache_gib=.5):
        super().__init__(bank, forward_batch_size, cache_gib, reserve_gib)
        self.features, self.images = OrderedDict(), OrderedDict()
        self.cpu_limit = int(cpu_cache_gib * 1024**3)
        self.image_limit = int(image_cache_gib * 1024**3)
        self.image_bytes = 0

    @staticmethod
    def nbytes(value):
        return value.numel() * value.element_size()

    def token_images(self, q):
        if q in self.images:
            self.images.move_to_end(q)
            return self.images[q]
        token = self.bank.tokens[q]
        value = torch.stack([self.bank.datasets[token['client_id']][j]['img'] for j in token['local_positions']])
        size = self.nbytes(value)
        while self.images and self.image_bytes + size > self.image_limit:
            _, previous = self.images.popitem(last=False)
            self.image_bytes -= self.nbytes(previous)
        if size <= self.image_limit:
            self.images[q] = value
            self.image_bytes += size
        return value

    def token_prefix(self, q, view):
        key = (q, view)
        if key in self.features:
            self.features.move_to_end(key)
            return self.features[key]
        chunks = []
        for image in self.token_images(q).split(self.bank.batch_size):
            image = image.to(self.bank.device)
            if view:
                image = image.flip(-1)
            chunks.append(self.prefix.forward_prefix(image.type(self.bank.core.dtype)))
        value = torch.cat(chunks).detach()
        size = self.nbytes(value)
        on_device = value.is_cuda and size <= self.cache_limit
        if on_device:
            free, _ = torch.cuda.mem_get_info(value.device)
            on_device = free >= self.reserve_bytes
        if not on_device:
            value = value.cpu()
        field = 'device_cache_bytes' if on_device else 'cpu_cache_bytes'
        limit = self.cache_limit if on_device else self.cpu_limit
        for old_key in list(self.features):
            if getattr(self, field) + size <= limit:
                break
            previous = self.features[old_key]
            if previous.is_cuda == on_device:
                del self.features[old_key]
                setattr(self, field, getattr(self, field) - self.nbytes(previous))
        if size <= limit:
            self.features[key] = value
            setattr(self, field, getattr(self, field) + size)
        return value
