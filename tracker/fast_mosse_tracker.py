import numpy as np
import pyfftw
import cv2
from collections import deque
from tracker.image_preprocessing_module import image_preprocessing
from tracker.synthetic_target import SyntheticTarget 
from tracker.kalman_filter import KalmanFilter

class FastMosseTracker:
    def __init__(self, 
                kalman: KalmanFilter=None,
                alpha_kalman_rate=1/30, 
                skip_frames=False, 
                max_skipped_frames=1, 
                debug=False, 
                alpha_smoothing=0.9, 
                training_images_count=9, 
                output_sigma_factor=0.05,
                correlation_target=0.3,
                correlation_target_k=2.2,
                rotation_range = (-3, 3),
                scale_range = (0.95, 1.05),
                scale_factors = [0.7, 0.85, 1.0],   
                is_recovery_enabled=False):
        self.is_debugging = debug
        self.current_roi = (0, 0, 0, 0)
        self.current_point = (0, 0)
        self.current_global_point = (0, 0)
        self.predicted_global_point = (0, 0)
        self.last_real_global_point = (0, 0)
        self.template_size = (0, 0)
        self.current_max_correlation = 0
        self.correlation_target = correlation_target
        self.min_correlation_for_update = correlation_target
        self.prev_correlation_target = correlation_target
        self.correlation_target_k = correlation_target_k
        self.training_images_count = training_images_count
        self.epsilon = 1e-6
        self.adaptive_rate = 0
        self.filter = None
        self.current_weight = None
        self.current_energy = None
        
        ## Synthetic target
        self.synthetic_target = None

        self.kalman_rate = alpha_kalman_rate
        ## Kalman Filter
        if kalman is None:
            self.kalman = KalmanFilter(self.kalman_rate)
        else:
            self.kalman = kalman
        self.predict = self.kalman.cv2_predict
        
        ## Tracking
        self.is_tracking = False
        self.tracking_lost_max_frames = 5
        self.tracking_lost_current_frame = 0
        self.d_max = 15
        self.correlation_alpha = 0.3
        
        
        ## Multiscale detection
        self.scale_factors = scale_factors
        self.current_template_scale = 1.0
        self.default_scale_frame_count = 5
        self.skip_multiscale_detection_frames = 3
        self.current_skipped_frames_multiscale_detection = 0
            
        
        ## Frame skipping
        self.skip_frames = skip_frames
        self.current_skipped_frame = 0
        self.max_skipped_frames = max_skipped_frames
        self.delay_first_frames = 10

        ## Training
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.output_sigma_factor = output_sigma_factor
        self.add_noise_and_illumination_changes = False
        self.add_illumination_changes = True
        self.add_noise_augmentation = True
    
        self.debug = None
        
        # Recovery
        self.is_recovery_enabled = is_recovery_enabled
        self.waiting_for_recovery = False
        self.orb = cv2.ORB_create(
            nfeatures = 10,
            scaleFactor = 1.05,
            nlevels=10,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=5
        )
        self.bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.sift = cv2.SIFT_create(
            #nfeatures = 10,
            #nOctaveLayers = 3,
            #contrastThreshold = 0.04,
            edgeThreshold = 5,
            #sigma = 1.9
        )
        self.bf_sift = cv2.BFMatcher()
        
        self.max_template_bank = 15
        self.template_bank = deque(maxlen=self.max_template_bank)
        
        # Point smoothness
        self.smoothed_x = None
        self.smoothed_y = None
        self.alpha_smoothing = alpha_smoothing
        
        #PyFFTW
        self.fft_object = None
        self.ifft_object = None
        self.input_buffer_fft = None
        self.output_buffer_fft = None
        self.input_buffer_ifft = None
        self.output_buffer_ifft = None
        self.num_pyfftw_threads = 0
        
        self.frame_count = 0
        
        
    def init(self, im, roi):
        x, y, w, h = roi
        h, w = self._ensure_even_dimensions(h, w)
        self.template_size = (h, w)
        self.template_size_base = self.template_size
        
        template = im[y:y+h, x:x+w]
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
        if self.is_recovery_enabled:
            kp, des = self.sift.detectAndCompute(template, None)
            self._add_data_to_template_bank((template, kp, des))
            
        self.input_buffer_fft = pyfftw.empty_aligned((h, w), dtype="complex64")
        self.output_buffer_fft = pyfftw.empty_aligned((h, w), dtype="complex64")
        self.input_buffer_ifft = self.output_buffer_fft
        self.output_buffer_ifft = pyfftw.empty_aligned((h, w), dtype="complex64")
        self.fft_object = pyfftw.FFTW(self.input_buffer_fft,
                                      self.output_buffer_fft,
                                      axes=(0, 1),
                                      direction="FFTW_FORWARD",
                                      flags=("FFTW_ESTIMATE",),
                                      threads=self.num_pyfftw_threads)
        self.ifft_object = pyfftw.FFTW(self.input_buffer_ifft,
                                      self.output_buffer_ifft,
                                      axes=(0, 1),
                                      direction="FFTW_BACKWARD",
                                      flags=("FFTW_ESTIMATE",),
                                      threads=self.num_pyfftw_threads)

        self.synthetic_target = SyntheticTarget()
        
        target = self.synthetic_target.make_synthetic_with_regularization(self.template_size[0], 
                                           self.template_size[1], 
                                           [(self.template_size[0] // 2, self.template_size[1] // 2)], output_sigma_factor=self.output_sigma_factor)
        synthetic_target_f = self._compute_fft(target)
        if self.add_noise_and_illumination_changes:
            training_images = self._generate_enhanced_training_images(template)
        else:
            training_images = self._generate_training_images(template)
 
        weight = np.zeros_like(target, dtype=np.complex64)
        energy = np.zeros_like(target, dtype=np.complex64)
        for image in training_images:
            preprocessed_image_f = self._compute_fft(image_preprocessing(image))
            weight += synthetic_target_f * preprocessed_image_f.conj()
            energy += preprocessed_image_f * preprocessed_image_f.conj()
            
        self.current_weight = weight
        self.current_energy = energy
        self.filter = weight / (energy + self.epsilon)
        
        #template_blurry = self._create_blurry_model(template)
        #template_processed = self._apply_model_subtraction(template, template_blurry)
        template_processed = self._process_template(template)
        result = self._compute_ifft(self.filter * self._compute_fft(image_preprocessing(template_processed)))
        self.current_max_correlation = np.max(result)
        self.current_point = (self.template_size[0] // 2, self.template_size[1] // 2)
        self.current_global_point = (int(self.current_point[0] + y), int(self.current_point[1] + x))

        self._smooth_current_point()

        self.predicted_global_point = self.predict(self.current_global_point[0], self.current_global_point[1])
        self.is_tracking = True     
        return True


    def update(self, im):
        if self.filter is None:
            raise Exception("Train the filter first!")

        self.frame_count += 1
        
        template, roi = self._extract_template(im, self.current_global_point, self.template_size)
        x = roi[0]
        y = roi[1]
        w = roi[2]
        h = roi[3]
        
        if self.skip_frames:
            if self.delay_first_frames > 0:
                self.delay_first_frames -= 1
            elif self.current_skipped_frame < self.max_skipped_frames and self.current_global_point != (0, 0):
                self.current_skipped_frame += 1
                self.current_global_point = self.predict(self.current_global_point[0], self.current_global_point[1])
                y = self.predicted_global_point[0] - self.template_size[0] // 2
                x = self.predicted_global_point[1] - self.template_size[1] // 2
                h = self.template_size[0]
                w = self.template_size[1]
                
                if self.is_debugging:
                    self.debug = {
                    "template": template,
                    "filter_result": None,
                    "target": None 
                    }

                self.current_roi = (x, y, w, h)
                return self.is_tracking, self.current_roi, self.debug
            else:
                self.current_skipped_frame = 0
        
        if self.frame_count > self.default_scale_frame_count:  
            if self.current_skipped_frames_multiscale_detection < self.skip_multiscale_detection_frames:
                self.current_skipped_frames_multiscale_detection += 1 
                new_point, best_roi, self.current_max_correlation, new_template_scale, correlation_map, preprocessed_template_f = self._multi_scale_detection(im, self.current_global_point, self.current_template_scale)
            else:
                new_point, best_roi, self.current_max_correlation, new_template_scale, correlation_map, preprocessed_template_f = self._multi_scale_detection(im, self.current_global_point)
                self.current_skipped_frames_multiscale_detection = 0
        else:
            #template_blurry = self._create_blurry_model(template)
            #template_processed = self._apply_model_subtraction(template, template_blurry)
            template_processed = self._process_template(template)
            preprocessed_template_f = self._compute_fft(image_preprocessing(template_processed))
            correlation_map = self._compute_ifft(self.filter * preprocessed_template_f)
            new_point = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
            best_roi = roi
            new_template_scale = 1.0
        
        dy = np.abs(self.current_point[0] - new_point[0])
        dx = np.abs(self.current_point[1] - new_point[1])
        
        if int(np.ceil(dy)) <= self.d_max and int(np.ceil(dx)) <= self.d_max:
            self.current_point = new_point
        else:
            self.current_point = (self.template_size[0] // 2, self.template_size[1] // 2)

        self._smooth_current_point()
        
        if self.is_recovery_enabled and self.current_max_correlation > 0.7:
            kp, des = self.sift.detectAndCompute(template, None)
            if des is not None:
                data = (template, kp, des.astype(np.float32))
                self._add_data_to_template_bank(data)

        target = self.synthetic_target.make_synthetic_with_regularization(h, w, [(self.current_point[0], self.current_point[1])])
        target_f = self._compute_fft(target)
           
        if self.current_max_correlation >= self.min_correlation_for_update:
            self.current_weight = self.adaptive_rate * (target_f * preprocessed_template_f.conj()) + (1.0 - self.adaptive_rate) * self.current_weight
            self.current_energy = self.adaptive_rate * (preprocessed_template_f * preprocessed_template_f.conj() + self.epsilon) + (1.0 - self.adaptive_rate) * self.current_energy
            self.filter = self.current_weight / (self.current_energy + self.epsilon) 
        
        correlation_map = self._compute_ifft(self.filter * preprocessed_template_f) 
        apce = self._calculate_apce(correlation_map, template)
        self._update_adaptive_learning_rate(apce)
        

        self._update_correlation_target(correlation_map, target)
        self.current_template_scale = new_template_scale
        
        if self.correlation_target < self.current_max_correlation:
            self.predicted_global_point = self.predict(self.current_global_point[0], self.current_global_point[1])
            self.is_tracking = True
        elif self.waiting_for_recovery and self.tracking_lost_current_frame < self.tracking_lost_max_frames:  
            self.tracking_lost_current_frame += 1
        else:
            self.is_tracking = False      

            self.tracking_lost_current_frame = 0
            
            if self.is_recovery_enabled:
                new_global_point, point_count = self._get_sift_point(im)

                if new_global_point != (0, 0):
                    if point_count == 1:
                        self.prev_correlation_target = self.correlation_target
                    else:
                        self.prev_correlation_target = 0.8 * self.correlation_target
                    self.current_global_point = new_global_point
                    self.waiting_for_recovery = False
             
        if self.is_tracking:
            self.current_global_point  = (roi[1] + int(self.current_point[0]), roi[0] + int(self.current_point[1]))
            self.last_real_global_point = self.current_global_point
        else:
            self.current_global_point = self.predicted_global_point
        self.current_roi = best_roi
        
        if self.is_debugging:   
            self.debug = {
                "template": template,
                "filter_result": correlation_map,
                "target": target,
                "adaptive_rate": self.adaptive_rate,
            } 
            
        return self.is_tracking, self.current_roi, self.debug
    
    
    def get_tracker_data(self):
        return {
            "roi": list(map(int, self.current_roi)),
            "correlation": float(self.current_max_correlation),
            "template_scale": float(self.current_template_scale),
            "learning_rate": float(self.adaptive_rate),
            "correlation_target": float(self.correlation_target)
        }
        
        
    def _compute_fft(self, image: np.ndarray) -> np.ndarray:
        self.input_buffer_fft.real[:] = image
        self.input_buffer_fft.imag[:] = 0.0
        self.fft_object()
        return self.output_buffer_fft.copy()
    
    
    def _compute_ifft(self, fft_spectrum: np.ndarray) -> np.ndarray:
        self.input_buffer_ifft[:] = fft_spectrum
        self.ifft_object()
        scaled_result = self.output_buffer_ifft
        return scaled_result.real.copy()
    
    def _calculate_apce(self, correlation_map: np.ndarray, template: np.ndarray) -> float:
        h, w = template.shape[:2]
        max_correlation = np.max(correlation_map)
        min_correlation = np.min(correlation_map)
        return (max_correlation - min_correlation)**2 / (np.sum((correlation_map - min_correlation)**2) / (h * w))
    
    def _calculate_background_stats(self, correlation_map: np.ndarray, target: np.ndarray) -> tuple:
        # Create binary mask for background
        peak_y, peak_x = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
        mask = np.ones_like(target, dtype=bool)
        radius = min(self.template_size) // 4  # Adjust radius based on template size
        y_grid, x_grid = np.ogrid[-peak_y:correlation_map.shape[0]-peak_y, -peak_x:correlation_map.shape[1]-peak_x]
        mask_area = x_grid*x_grid + y_grid*y_grid <= radius*radius
        mask[mask_area] = False
        
        # Calculate background statistics
        background_values = correlation_map[mask]
        bg_mean = np.mean(background_values)
        bg_std = np.std(background_values)
        
        return bg_mean, bg_std
    
    
    # def _update_correlation_target1(self, correlation_map, target):
    #     bg_mean, bg_std = self._calculate_background_stats(correlation_map, target)
    #     current_adaptive_threshold = bg_mean + self.correlation_target_k * bg_std
    #     self.correlation_target = (1 - self.correlation_alpha) * self.prev_correlation_target + self.correlation_alpha * current_adaptive_threshold
    #     self.prev_correlation_target = self.correlation_target
        
    def _update_correlation_target(self, correlation_map, target):
        correlation_map_bg = correlation_map - target 
        current_adaptive_threshold = np.mean(correlation_map_bg) + self.correlation_target_k * np.std(correlation_map_bg)
        self.correlation_target = (1 - self.correlation_alpha) * self.prev_correlation_target + self.correlation_alpha * current_adaptive_threshold
        self.prev_correlation_target = self.correlation_target
        
        
    def _update_adaptive_learning_rate1(self, apce):
        # Base learning rate parameters
        min_rate = 0.001  # Minimum learning rate
        max_rate = 0.15    # Maximum learning rate
        mid_point = 25     # APCE value at middle of range
        steepness = 0.1    # Controls how quickly rate changes
        
        # Sigmoid-like function for smooth transition
        self.adaptive_rate = min_rate + (max_rate - min_rate) * (
            1 / (1 + np.exp(-steepness * (apce - mid_point)))
        )
        
        # Ensure rate stays within bounds
        self.adaptive_rate = np.clip(self.adaptive_rate, min_rate, max_rate)
    
    
    def _update_adaptive_learning_rate(self, apce):
        if apce > 50:
            self.adaptive_rate = 0.15
        elif apce > 40:
            self.adaptive_rate = 0.125
        elif apce > 30:
            self.adaptive_rate = 0.1
        elif apce > 25:
            self.adaptive_rate = 0.065
        elif apce > 20:
            self.adaptive_rate = 0.0475
        elif apce > 15:
            self.adaptive_rate = 0.0275
        elif apce > 10:
            self.adaptive_rate = 0.0225
        elif apce > 5:
            self.adaptive_rate = 0.0175
        else:
            self.adaptive_rate = 0.001
    
    
    def _smooth_current_point(self):
        if self.smoothed_x is None:
            self.smoothed_x = self.current_point[1]
            self.smoothed_y = self.current_point[0]
        else:
            self.smoothed_x = (1 - self.alpha_smoothing) * self.smoothed_x + self.alpha_smoothing * self.current_point[1]
            self.smoothed_y = (1 - self.alpha_smoothing) * self.smoothed_y + self.alpha_smoothing * self.current_point[0]

        self.current_point = (self.smoothed_y, self.smoothed_x)
        

    
    # def _smooth_current_point1(self):
    #     if self.current_max_correlation > 0.8:
    #         alpha = 0.9 
    #     elif self.current_max_correlation > 0.6:
    #         alpha = 0.7
    #     else:
    #         alpha = 0.5 

    #     if self.smoothed_x is None:
    #         self.smoothed_x = self.current_point[1]
    #         self.smoothed_y = self.current_point[0]
    #     else:
    #         self.smoothed_x = (1 - alpha) * self.smoothed_x + alpha * self.current_point[1]
    #         self.smoothed_y = (1 - alpha) * self.smoothed_y + alpha * self.current_point[0]

    #     self.current_point = (self.smoothed_y, self.smoothed_x)


    def _multi_scale_detection(self, image, center, current_scale=None):
        best_correlation = -1
        best_roi = None
        best_scale = 1.0
        best_correlation_map = None
        best_point = (0, 0)
        best_template = None
        
        h_base, w_base = self.template_size_base

        
        if current_scale is not None:
            scale_factors = [current_scale]
        else:
            if self.current_max_correlation > 0.7:
                scale_factors = [self.current_template_scale] 
            else:
                scale_factors = self.scale_factors

        #scale_factors = self.scale_factors
        for scale in scale_factors:
            
            # Calculate scaled template size
            h_scaled = int(h_base * scale)
            w_scaled = int(w_base * scale)
            
            h_scaled, w_scaled = self._ensure_even_dimensions(h_scaled, w_scaled)
            
            # Extract template at this scale
            template, roi = self._extract_template(image, center, (h_scaled, w_scaled))
            
            if template.size == 0:
                continue

            template = cv2.resize(template, (w_base, h_base), interpolation=cv2.INTER_LINEAR)
            #template_blurry = self._create_blurry_model(template)
            #template_processed = self._apply_model_subtraction(template, template_blurry)
            template_processed = self._process_template(template)
            
            preprocessed_template_f = self._compute_fft(image_preprocessing(template_processed)) 
            correlation_map = self._compute_ifft(self.filter * preprocessed_template_f)
            max_correlation = np.max(correlation_map)
            
            if max_correlation > best_correlation: 
                best_roi = roi
                best_scale = scale
                best_template = preprocessed_template_f
                best_correlation = max_correlation
                best_correlation_map = correlation_map
                
        if best_correlation_map is not None:
            best_point = np.unravel_index(np.argmax(best_correlation_map), best_correlation_map.shape)
        else:
            best_point = (h_base // 2, w_base // 2)
  
        return best_point, best_roi, best_correlation, best_scale, best_correlation_map, best_template

    
    def _get_orb_point(self, image: np.ndarray) -> tuple:
        size_mul = 3
        bigger_template_size = (int(self.template_size[0] * size_mul), int(self.template_size[1] * size_mul))
        big_template, big_roi = self._extract_template(image, self.current_global_point, bigger_template_size)

        kp_big = self.orb.detect(big_template, None)
        kp_big, des_big = self.orb.compute(big_template, kp_big)
        for template_data in self.template_bank:        
            template, kp, des = template_data
            matches = self.bf_orb.match(des, des_big)
            matches = sorted(matches, key = lambda x: x.distance)
            
            if len(matches) == 0:
                return (0, 0)    
                
            x_sum = 0
            y_sum = 0
            for match in matches:
   
                x_sum += kp_big[match.trainIdx].pt[0]
                y_sum += kp_big[match.trainIdx].pt[1]
            
            new_global_point = (big_roi[1] + int(y_sum / len(matches)), big_roi[0] + int(x_sum / len(matches)))
            
            #matches_im = cv2.drawMatches(template,kp,big_template,kp_big,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #plt.imshow(matches_im)
            #plt.show()
            return new_global_point
   
    def _get_sift_point(self, image: np.ndarray) -> tuple:
        size_mul = 2.5
        bigger_template_size = (int(self.template_size[0] * size_mul), int(self.template_size[1] * size_mul))
        big_template, big_roi = self._extract_template(image, self.current_global_point, bigger_template_size)

        kp_big, des_big = self.sift.detectAndCompute(big_template, None)
        
        default = ((0, 0), 0)
        
        if des_big is None:
            return default
        
        for template_data in self.template_bank:        
            template, kp, des = template_data
            
            
            matches = self.bf_sift.knnMatch(des, des_big.astype(np.float32), k = 2)
            
            if len(matches) == 0 or len(matches[0]) < 2:
                continue
            
            ratio_threshold = 0.7
            good = []
            for m,n in matches:
                if m.distance < ratio_threshold * n.distance:
                    good.append([m])    

            if len(good) == 0:
                return default

            x_sum = 0
            y_sum = 0
            for match in good:
   
                x_sum += kp_big[match[0].trainIdx].pt[0]
                y_sum += kp_big[match[0].trainIdx].pt[1]
            
            new_global_point = (big_roi[1] + int(y_sum / len(good)), big_roi[0] + int(x_sum / len(good)))

            # new_global_point = (int(big_roi[1] + kp_big[0].pt[1]), int(big_roi[0] + kp_big[0].pt[0]))
            
            
            #matches_im = cv2.drawMatchesKnn(template,kp,big_template,kp_big,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #plt.imshow(matches_im)
            #plt.show()
            # x1 = new_global_point[1] - self.template_size[1] // 2
            # y1 = new_global_point[0] - self.template_size[0] // 2
            # x2 = new_global_point[1] + self.template_size[1] // 2
            # y2 = new_global_point[0] + self.template_size[0] // 2
            
            #new_gp_im = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 120, 120), 2)
            #plt.imshow(new_gp_im)
            #plt.show()
            return new_global_point, len(good)
        return default

    def _add_data_to_template_bank(self, data) -> None:
        # Use collections.deque with maxlen instead of manual list management
        self.template_bank.append(data)
    
    # def _add_data_to_template_bank(self, data) -> None:
    #     self.template_bank.append(data)
    #     if len(self.template_bank) > self.max_template_bank:
    #         self.template_bank.pop(0)
        
        
    # def _extract_template1(self, image: np.ndarray, center: tuple, size: tuple) -> tuple:
    #     h, w = size
    #     cy, cx = center
        
    #     x1 = max(0, cx - w // 2)
    #     y1 = max(0, cy - h // 2)
    #     x2 = min(image.shape[1], x1 + w)
    #     y2 = min(image.shape[0], y1 + h)
        
    #     if x2 - x1 != w:
    #         x1 = max(0, x2 - w)
    #     if y2 - y1 != h:
    #         y1 = max(0, y2 - h)
            
    #     template = image[y1:y2, x1:x2]
        
    #     if len(template.shape) == 3:
    #         template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
    #     return template, (x1, y1, x2 - x1, y2 - y1)
    
    
    def _extract_template(self, image: np.ndarray, center: tuple, size: tuple) -> tuple:
        h, w = size
        cy, cx = center
        x1 = np.clip(cx - w // 2, 0, image.shape[1] - w)
        y1 = np.clip(cy - h // 2, 0, image.shape[0] - h)
        template = image[y1:y1 + h, x1:x1 + w]
        return cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template, (x1, y1, w, h)
    
    
    def _ensure_even_dimensions(self, h: int, w: int) -> tuple:
        return (h + 1 if h % 2 != 0 else h, w + 1 if w % 2 != 0 else w)
    
    
    def _generate_training_images(self, template: np.ndarray) -> list:

        if self.training_images_count <= 1:
            return [template]
            
        training_images = [template]
        h, w = template.shape
        center = (w // 2, h // 2)
        
        remaining_count = self.training_images_count - 1
        rotation_count = remaining_count // 2
        scale_count = remaining_count - rotation_count

        if rotation_count > 0:
            angles = np.linspace(self.rotation_range[0], self.rotation_range[1], rotation_count)
            for angle in angles:
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    template, rotation_matrix, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_TRANSPARENT
                )
                training_images.append(rotated)

        if scale_count > 0:
            scales = np.linspace(self.scale_range[0], self.scale_range[1], scale_count)
            for scale in scales:
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                scaled = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                if scale > 1.0:
                    start_x = (new_w - w) // 2
                    start_y = (new_h - h) // 2
                    cropped = scaled[start_y:start_y + h, start_x:start_x + w]
                    training_images.append(cropped)       
                else:
                    pad_x = (w - new_w) // 2
                    pad_y = (h - new_h) // 2
                    padded = cv2.copyMakeBorder(
                        scaled, pad_y, h - new_h - pad_y, pad_x, w - new_w - pad_x,
                        cv2.BORDER_ISOLATED
                    )
                    training_images.append(padded)
        
        return training_images
    
    
    def _generate_enhanced_training_images(self, template: np.ndarray) -> list:
        if self.training_images_count <= 1:
            return [template]
            
        training_images = [template] 
        h, w = template.shape
        center = (w // 2, h // 2)
        
        remaining_count = self.training_images_count - 1
        rotation_count = remaining_count // 3
        scale_count = remaining_count // 3
        noise_count = remaining_count - rotation_count - scale_count
        
        if rotation_count > 0:
            angles = np.linspace(self.rotation_range[0], self.rotation_range[1], rotation_count)
            for angle in angles:
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    template, rotation_matrix, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT_101
                )
                training_images.append(rotated)
        
        if scale_count > 0:
            scales = np.linspace(self.scale_range[0], self.scale_range[1], scale_count)
            for scale in scales:
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                scaled = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                if scale > 1.0:
                    start_x = (new_w - w) // 2
                    start_y = (new_h - h) // 2
                    cropped = scaled[start_y:start_y + h, start_x:start_x + w]
                    training_images.append(cropped)       
                else:
                    pad_x = (w - new_w) // 2
                    pad_y = (h - new_h) // 2
                    padded = cv2.copyMakeBorder(
                        scaled, pad_y, h - new_h - pad_y, pad_x, w - new_w - pad_x,
                        cv2.BORDER_REFLECT_101
                    )
                    training_images.append(padded)
        
        if noise_count > 0:
            for i in range(noise_count):
                
                result = template.astype(np.float32)
                # Illumination changes
                if self.add_illumination_changes:
                    # Gamma correction variation
                    gamma = np.random.uniform(0.8, 1.2)
                    result = np.power(result / 255.0, gamma) * 255.0
                    
                    # Brightness/contrast variation
                    alpha = np.random.uniform(0.9, 1.1)  # Contrast
                    beta = np.random.uniform(-10, 10)    # Brightness
                    result = alpha * result + beta
                
                # Gaussian noise
                if self.add_noise_augmentation:
                    noise_std = np.random.uniform(1, 5)
                    noise = np.random.normal(0, noise_std, template.shape)
                    result = result + noise
                
                augmented = np.clip(result, 0, 255).astype(np.uint8)
                training_images.append(augmented)
                
        return training_images
    
    
    def _create_blurry_model(self, template, gaussian_sigma_factor=0.0475):
        
        template_float = template.astype(np.float32)
        gaussian_sigma = np.max(template_float.shape) * gaussian_sigma_factor
        
        if gaussian_sigma < 1.0:
            gaussian_sigma = 1
        
        background_model = cv2.GaussianBlur(template_float, (0, 0), sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)
        return background_model
    
    
    def _apply_model_subtraction(self, image, model):
        if model is None:
            return image
        
        foreground = image.astype(np.float32) - model.astype(np.float32)
        
        fg_min, fg_max = np.min(foreground), np.max(foreground)
        if fg_max - fg_min > 0:
            foreground = ((foreground - fg_min) / (fg_max - fg_min)) * 255.0

        return foreground.astype(np.float32)
    
    def _process_template(self, template, gaussian_sigma_factor=0.0475):
        template_float = template.astype(np.float32)
        
        gaussian_sigma = max(1.0, np.max(template_float.shape) * gaussian_sigma_factor)
        
        background_model = cv2.GaussianBlur(template_float, (0, 0), 
                                        sigmaX=gaussian_sigma, 
                                        sigmaY=gaussian_sigma)
        
        cv2.subtract(template_float, background_model, template_float)
        
        cv2.normalize(template_float, template_float, 0, 255, cv2.NORM_MINMAX)
        
        return template_float