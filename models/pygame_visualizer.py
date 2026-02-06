import pygame

class PygameVisualizer():
	def __init__(self, model, width=800, height=600, title="Pygame Visualizer"):
		self.model = model
		self.width = width
		self.height = height
		self.title = title
		pygame.init()
		self.screen = pygame.display.set_mode((self.width, self.height))
		pygame.display.set_caption(self.title)
		self.clock = pygame.time.Clock()
		self.running = True

	def draw(self):
		"""Draw the current state to the screen. Must be implemented by child classes."""
		pass
	
	def handle_events(self, events):
		pass

	def update(self):
		"""Update the model or visualization state. Must be implemented by child classes."""
		pass

	def run(self, fps=60):
		while self.running:
			self.handle_events(pygame.event.get())
			self.update()
			self.draw()
			pygame.display.flip()
			self.clock.tick(fps)
		pygame.quit()
