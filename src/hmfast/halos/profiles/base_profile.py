from abc import ABC, abstractmethod


class HaloProfile(ABC):
    """
    Grandparent halo profile class from which all halo profile classes inherit.

    Child profile classes must implement :meth:`real` and :meth:`fourier`. A profile without a closed-form
    Fourier transform can sample :meth:`real` on a log-spaced grid and pass it to
    :func:`hmfast.utils.hankel_transform`.
    """

    @abstractmethod
    def real(self, halo_model, r, m, z):
        pass

    @abstractmethod
    def fourier(self, halo_model, k, m, z):
        pass
