import numpy as np

from typing import Callable


class SinusoidalImage:

	def __init__(self,
			  wawelength: float,
			  amplitude: Callable[[float, float], float],
			  phase: Callable[[float], float]) -> None:
		
		self.p = wawelength
		self.A = amplitude
		self.phase = phase

	def __signal(self, x: float, y: float) -> float:
		return self.A(x,y) * np.cos((2*np.pi/self.p) * y + self.phase(y))
	
	@property
	def signal(self) -> Callable[[float, float], float]:
		return np.vectorize(self.__signal)
	
	def generate(self, x_min: float, x_max: float, y_min: float, y_max: float,
		  Nx: int, Ny: int) -> np.ndarray:

		x = np.linspace(start=x_min, stop=x_max, num=Nx)
		y = np.linspace(start=y_min, stop=y_max, num=Ny)
		X, Y = np.meshgrid(x,y)

		return self.signal(X,Y) # type: ignore
