import numpy as np
from astropy.modeling import fitting, models
import matplotlib.pyplot as plt

y, x = np.mgrid[-1000:1001, -1000:1001]


def elliptical_transform(fake_sigx, fake_sigy, theta):
    def sigma(a, b, m):
        sigma = np.sqrt(np.abs(np.divide((1 + m ** 2), ((1 / a) ** 2 + (m / b) ** 2))))
        return sigma

    m_x = -np.tan(theta)
    m_y = np.tan(np.pi / 2 - theta)
    true_sigx = sigma(fake_sigx, fake_sigy, m_x)
    true_sigy = sigma(fake_sigx, fake_sigy, m_y)
    return true_sigx, true_sigy


model = models.Gaussian2D(
    x_mean=0, y_mean=0, x_stddev=500, y_stddev=200, theta=np.pi / 2
)
plt.imshow(model(x, y))
print(model.x_stddev)
print(model.y_stddev)
print(model.theta)
true_sigx, true_sigy = elliptical_transform(500, 200, np.pi / 2)
print(true_sigx)
print(true_sigy)
