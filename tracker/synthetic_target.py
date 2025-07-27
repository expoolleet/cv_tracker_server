import numpy as np

class SyntheticTarget:
    # def __init__(self, height=0, width=0, output_sigma_factor = 0.05):
    #     self.height = height
    #     self.width = width
    #     self.output_sigma_factor = output_sigma_factor
    #     self.sigma = np.sqrt(height * width) * output_sigma_factor
    
    
    def make_synthetic(self, height, width, target_positions, output_sigma_factor=0.05):  
        y_grid, x_grid = np.mgrid[0:height,0:width]

        self.sigma = np.sqrt(height * width) * output_sigma_factor#self.output_sigma_factor

        g = np.zeros((height, width))
        for y_j, x_j in target_positions:
            L2 = (y_grid - y_j)**2 + (x_grid - x_j)**2
            g += np.exp(-0.5 * L2 / (self.sigma**2))
                    
        g = g / np.max(g + 1e-6)
        return g
    
    
    def make_synthetic_with_regularization(self, height, width, target_positions, 
                                         regularization_lambda=0.01, output_sigma_factor=0.05):
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        
        self.sigma = np.sqrt(height * width) * output_sigma_factor#self.output_sigma_factor
        
        g = np.zeros((height, width))
        for y_j, x_j in target_positions:
            L2 = (y_grid - y_j)**2 + (x_grid - x_j)**2
            g += np.exp(-L2 / (2 * self.sigma**2))
        
        # Normalize
        g = g / (np.max(g) + 1e-6)
        
        # Add regularization to prevent overfitting
        if regularization_lambda > 0:
            # Add small amount of uniform distribution
            uniform_component = regularization_lambda * np.ones_like(g) / (height * width)
            g = (1 - regularization_lambda) * g + uniform_component
        
        return g