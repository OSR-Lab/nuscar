# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

from tqdm.notebook import tqdm
import numpy as _np
import pandas as _pd
from IPython.display import HTML
import nuscar
import logging
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


SUPPORTED_FRAME_TYPES = (slice,
                         int, list, _np.ndarray, range)


class Container():
    """Base class for a container to fetch traces from trace file.
    """

    def __init__(self, frame=None, func_preprocess=None):
        self._func_pre = func_preprocess
        self.logger = logging.getLogger(__name__)
        self._check_frame_type(frame)
        self._frame = frame

    def _check_frame_type(self, frame):
        if frame is None:
            return
        if not isinstance(frame, SUPPORTED_FRAME_TYPES):
            raise IndexError(
                "frame type can only be {types}".format(
                    types=", ".join([str(t)
                                    for t in SUPPORTED_FRAME_TYPES]),
                )
            )

    def __dir__(self):
        # Tab completion in jupyter
        return self.metadatas + ['samples', 'metadatas', 'view', 'viewmeta', 'viewxyz', 'viewspectrogram', 'check_sync_func']

    def __getattr__(self, name):
        pass

    def __getitem__(self, key):
        pass

    @ property
    def metadatas(self):
        pass

    @property
    def _samples(self) -> _np.ndarray:
        """rewrite in clild class to get samples."""
        pass

    @property
    def _nb_traces(self) -> int:
        """rewrite in clild class to get trace number."""
        pass

    @property
    def samples(self):
        smp = self._samples
        smp = smp.reshape(-1, smp.shape[-1])
        if self._frame is not None:
            smp = smp[:, self._frame]
        if self._func_pre is not None:
            smp = self._func_pre(smp)
        return smp

    def _repr_html_(self):
        info = f"""
        <table>
            <tr>
                <th width="100px">Trace Count</th>
                <th width="100px">Sample Points</th>
                <th width="200px">Meta Info</th>
            </tr>
            <tr>
                <td>%d</td>
                <td>%d</td>
                <td>%s</td>
            </tr>
        </table>
        """ % (self._nb_traces, self[0].samples.shape[-1], str(self.metadatas))
        return '<div>'+info+'</div>'

    def check_sync_func(self, sync_func, check_number=3):
        index = _np.random.randint(low=self._nb_traces, size=check_number)
        samples = self[index].samples
        l = []
        x = []
        for i, s in enumerate(samples):
            t = sync_func(s)
            if t is not None:
                l.append(t)
                x.append(str(index[i]))
            else:
                self.logger.info(f"Discarding trace [%d]." % index[i])
        if len(l) > 0:
            s = _np.vstack(l)
            return nuscar.plot(s, name=x)

    def view(self, index=range(3), webgl=True, resample='auto', shift=False):
        if isinstance(index, int):
            index = [index]
        else:
            index = list(index)
        if len(self) < len(index):
            index = list(range(len(self)))
        samples = self[index].samples
        name = [str(i) for i in index]
        return nuscar.plot(samples, name=name, webgl=webgl, resample=resample, shift=shift)
    
    def viewspectrogram(self, index=0, fs=1.0, N=256, fslim=None):
        index = [index]
        samples = _np.squeeze(self[index].samples)
        return nuscar.plot_spectrogram(samples, fs=fs, N=N, fslim=fslim)

    def viewxyz(self, resample=False, yzoom=False, positionmeta="position", method="mean"):
        pos = self.__getattr__(positionmeta)
        samples = self.samples
        nuscar.plot_heatmap(pos, samples, resample=resample, yzoom=yzoom, method=method)
    
    def viewmeta(self, index=range(5)):
        if isinstance(index, int):
            index = [index]
        index = list(index)
        df = _pd.DataFrame({'No.': index})
        for m in self.metadatas:
            d = self[index].__getattr__(m)
            df[m] = _pd.Series(list(d.reshape(-1, d.shape[-1])))
        return HTML(df.to_html(index=False))


class Storer:
    """Base class to store a trace file.
    """

    def __init__(self):
        self._meta = None
        self.logger = logging.getLogger(__name__)

    def update(self, samples: _np.array, **kwargs):
        if self._meta is None:
            self._meta = list(kwargs)
        elif not self._meta == list(kwargs):
            self.logger.debug(self._meta, list(kwargs))
            raise KeyError("mismatch meta info.")
        else:
            pass

    def update_container(self, ctn: Container, metakeep=None):
        """Update storer from a opened container"""
        nb_traces = len(ctn)
        nb_samples = len(ctn[0].samples)
        batch_size = nuscar._find_batch_size(nb_samples)
        pbar = tqdm(total=nb_traces)
        for i in range(0, nb_traces, batch_size):
            batch = ctn[i:i+batch_size]
            meta = dict()
            if metakeep is None:
                for m in ctn.metadatas:
                    meta[m] = getattr(batch, m)
            else:
                for m in metakeep:
                    meta[m] = getattr(batch, m)
            self.update(samples=batch.samples, **meta)
            pbar.update(len(batch))
        pbar.close()

    def update_sync(self, ctn: Container,  sync_func: callable, metakeep: list = None):
        """Use a function sync_func to synchronize a traceset.

        Args:
            ctn (Container): container of unsynchronized traces.
            sync_func (callable): sync_func(t:ndarray) -> ndarray, where s and return are 1-D ndarray.
            metakeep (list, optional): meta to keep in storer. Defaults to None(keep all).
        """
        nb_traces = len(ctn)
        nb_samples = len(ctn[0].samples)
        nb_samples_sync = 0
        for c in ctn:
            s = sync_func(_np.squeeze(c.samples))
            if s is not None:
                nb_samples_sync = len(s)
                self.logger.debug('number sync samples = ', nb_samples_sync)
                sync_type = s.dtype
                break
        if nb_samples_sync == 0:
            raise RuntimeError("sync_func always returns None, cannot perform alignment")

        batch_size = nuscar._find_batch_size(nb_samples)
        pbar = tqdm(total=nb_traces)
        keep_number = 0
        for i in range(0, nb_traces, batch_size):
            batch = ctn[i:i+batch_size]
            keep_idx = []
            keep_smp = _np.empty(
                [batch_size, nb_samples_sync], dtype=sync_type)
            idx = 0
            for j, t in enumerate(batch.samples):
                s = sync_func(t)
                if s is not None:
                    keep_smp[idx] = s
                    keep_idx.append(j)
                    idx = idx+1
                else:
                    self.logger.debug("Discarding trace %d" % (i+j))
            if idx > 0:
                meta = dict()
                batch_keep = batch[keep_idx]
                if metakeep is None:
                    for m in ctn.metadatas:
                        meta[m] = getattr(batch_keep, m)
                else:
                    for m in metakeep:
                        meta[m] = getattr(batch_keep, m)
                self.update(samples=keep_smp[:idx], **meta)
            keep_number = keep_number+idx
            pbar.update(len(batch))
        pbar.close()
        self.logger.info(f"Kept [%d] traces, success rate [%.2f%%]" % (
            keep_number, float(keep_number * 100) / nb_traces))

    def close(self):
        """rewrite in clild class."""
        pass
