import cv2
import numpy as np

class KalmanFilter:

    def __init__(self, dt, max_k = 5):
        self.k = 0
        self.max_k = max_k
        self.dt = dt
        
        self.x = 0
        self.x_previous = 0
        self.x_predicted = 0
        
        self.vx_predicted = 0
        self.vx_previous = 0
        
        self.ax_predicted = 0
        self.ax_previous = 0
        
        self.y = 0
        self.y_previous = 0
        self.y_predicted = 0
        
        self.vy_predicted = 0
        self.vy_previous = 0
        
        self.ay_predicted = 0
        self.ay_previous = 0
        
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0], 
            [0, 1, 0, 0]], np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0], 
            [0, 1, 0, 1], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]], np.float32)
        
        self.kf.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], np.float32) * 0.5
        
        self.kf.measurementNoiseCov = np.array([
            [1, 0],
            [0, 1]], np.float32) * 0.9
        
        
    def reset(self):
        self.k = 0
    
    def alpha_beta_predict(self, y, x):
        if self.k == 0:          
            self.x_previous = x
            self.y_previous = y
            self.k = 1
            return (y, x)
        
        if self.k == 1:
            x_prev = self.x_previous
            y_prev = self.y_previous
            self.x_previous = x
            self.y_previous = y
            self.vx_previous = (x - x_prev) / self.dt
            self.vy_previous = (y - y_prev) / self.dt  
            self.ax_previous = 0
            self.ay_previous = 0
            self.k = 2
            return (y, x)
        
        alpha = 2. * (2. * self.k - 1.) / (self.k * (self.k + 1.))
        beta = 6. / (self.k * (self.k + 1.))
        
        self.x_predicted = self.x_previous + self.vx_previous * self.dt
        self.y_predicted = self.y_previous + self.vy_previous * self.dt
        
        self.vx_predicted = self.vx_previous
        self.vy_predicted = self.vy_previous
        
        self.x = int(self.x_predicted + alpha * (x - self.x_predicted))
        self.y = int(self.y_predicted + alpha * (y - self.y_predicted))
        
        beta_dt = beta / self.dt
        self.vx_previous = self.vx_predicted + beta_dt * (x - self.x_predicted)
        self.vy_previous = self.vy_predicted + beta_dt * (y - self.y_predicted)
        
        self.x_previous = self.x
        self.y_previous = self.y
        
        if self.k < self.max_k:  
            self.k += 1
        else:
            self.k = self.max_k
            
        return (self.y, self.x)
    
    
    def alpha_beta_gamma_predict(self, y, x):
        if self.k == 0:          
            self.x_previous = x
            self.y_previous = y
            self.k = 1
            return y, x
        
        if self.k == 1:
            x_prev = self.x_previous
            y_prev = self.y_previous
            self.x_previous = x
            self.y_previous = y
            self.vx_previous = (x - x_prev) / self.dt
            self.vy_previous = (y - y_prev) / self.dt  
            self.ax_previous = ((x - x_prev) / self.dt - self.vx_previous / self.dt) 
            self.ay_previous = ((y - y_prev) / self.dt - self.vy_previous / self.dt) 
            self.k = 2
            return y, x
        
        alpha = 2. * (2. * self.k - 1.) / (self.k * (self.k + 1.))
        beta = 6. / (self.k * (self.k + 1.))
        gamma = 12. / (self.k * (self.k + 1) * (self.k + 2))
        
        self.x_predicted = self.x_previous + self.vx_previous * self.dt + 0.5 * self.ax_previous * self.dt**2
        self.y_predicted = self.y_previous + self.vy_previous * self.dt + 0.5 * self.ay_previous * self.dt**2
        
        self.vx_predicted = self.vx_previous + self.ax_previous * self.dt
        self.vy_predicted = self.vy_previous + self.ay_previous * self.dt
        
        self.ax_predicted = self.ax_previous
        self.ay_predicted = self.ay_previous
        
        self.x = int(self.x_predicted + alpha * (x - self.x_predicted))
        self.y = int(self.y_predicted + alpha * (y - self.y_predicted))
        
        beta_dt = beta / self.dt
        self.vx_previous = self.vx_predicted + beta_dt * (x - self.x_predicted)
        self.vy_previous = self.vy_predicted + beta_dt * (y - self.y_predicted)
        
        gamma_dt = gamma / self.dt**2
        self.ax_previous = self.ax_predicted + gamma_dt * (x - self.x_predicted)
        self.ay_previous = self.ay_predicted + gamma_dt * (y - self.y_predicted)
        
        self.x_previous = self.x
        self.y_previous = self.y
        
        if self.k < self.max_k:  
            self.k += 1
        else:
            self.k = self.max_k
            
        return self.y, self.x


    def cv2_predict(self, coordY, coordX):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        self.x, self.y = int(predicted[0]), int(predicted[1])
        return self.y, self.x
    
    
    def cv2_predict1(self, coordY, coordX, measurement_available=True):
        """
        Выполняет шаги предсказания и коррекции фильтра Калмана.

        Args:
            coordY (float): Измеренная Y-координата (от MOSSE/SIFT).
            coordX (float): Измеренная X-координата (от MOSSE/SIFT).
            measurement_available (bool): True, если есть надежное измерение от трекера,
                                          иначе фильтр только предсказывает.

        Returns:
            tuple: (smoothed_y, smoothed_x) - сглаженное положение объекта.
        """
        # ШАГ 1: ПРЕДСКАЗАНИЕ
        # Предсказываем следующее состояние на основе текущего (предыдущего скорректированного) состояния
        predicted_state = self.kf.predict()
        
        # Если нужно использовать предсказанное для ROI следующего поиска, то вот оно:
        predicted_x_for_roi = predicted_state[0, 0]
        predicted_y_for_roi = predicted_state[1, 0]

        # ШАГ 2: КОРРЕКЦИЯ
        if measurement_available:
            measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
            # Корректируем предсказанное состояние с помощью нового измерения
            corrected_state = self.kf.correct(measured)
        else:
            # Если измерение недоступно или ненадежно (например, низкий APCE),
            # мы используем только предсказанное состояние.
            corrected_state = predicted_state # Фактически, не корректируем, а принимаем предсказание

        # Извлекаем скорректированные (сглаженные) X и Y
        self.x = int(corrected_state[0, 0])
        self.y = int(corrected_state[1, 0])

        return self.y, self.x

