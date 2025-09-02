cimport numpy as np
import numpy as np
np.import_array()

import pyfftw
import cv2

from collections import deque

from tracker.image_preprocessing_module import image_preprocessing

from tracker.synthetic_target import make_synthetic_with_regularization 
from tracker.kalman_filter import KalmanFilter
from tracker.xor_tracker import XORTracker

COMPLEX_TYPE = np.complex64
complex_type_name = "complex64"
ctypedef np.complex64_t COMPLEX_TYPE_t

FLOAT_TYPE = np.float32
ctypedef np.float32_t FLOAT_TYPE_t

cdef extern from "stdbool.h":
    ctypedef int bool

cdef class FastMosseTracker:
    
    cdef:
        dict debug

        tuple current_roi
        tuple current_point # (y, x)
        tuple current_global_point # (y, x)
        tuple predicted_global_point
        tuple last_real_global_point
        tuple template_size
        tuple template_size_base
        tuple rotation_range
        tuple scale_range

        readonly FLOAT_TYPE_t last_max_correlation
        readonly FLOAT_TYPE_t correlation_target
        readonly FLOAT_TYPE_t min_correlation_for_update
        readonly FLOAT_TYPE_t adaptive_rate
        FLOAT_TYPE_t epsilon
        FLOAT_TYPE_t smoothed_x
        FLOAT_TYPE_t smoothed_y
        FLOAT_TYPE_t alpha_smoothing
        FLOAT_TYPE_t output_sigma_factor
        FLOAT_TYPE_t kalman_rate
        FLOAT_TYPE_t current_template_scale
        FLOAT_TYPE_t multi_scale_detection_good_correlation
        FLOAT_TYPE_t high_correlation_threshold
        FLOAT_TYPE_t correlation_deviation

        np.ndarray current_weight
        np.ndarray current_energy
        np.ndarray computed_filter
        np.ndarray init_frame
        np.ndarray init_template
        np.ndarray init_template_f
        np.ndarray high_correlation_frame
        np.ndarray high_correlation_template
        np.ndarray high_correlation_template_f
        np.ndarray centric_synthetic_target
        np.ndarray centric_synthetic_target_f

        tuple original_roi
        tuple high_correlation_template_roi

        list scale_factors
        list expanded_scale_factors
        list correlation_history
        list tracker_bank
        list search_strategies

        bool is_debugging
        bool is_tracking
        bool skip_frames
        bool is_recovery_enabled
        bool is_tracking_recovered
    
        int training_images_count
        int tracking_lost_max_attempts
        int tracking_lost_current_attempt
        int d_max
        int default_scale_frame_count
        int skip_multiscale_detection_frames
        int current_skipped_frames_multiscale_detection
        int skipped_frame_count
        int max_skipped_frames
        int delay_first_frames
        int tracker_bank_max_count
        int num_pyfftw_threads
        int low_correlation_frame_count
        int low_correlation_frame_count_max
        int tracker_bank_index
        int frame_count_from_last_recovery
        int frame_count
        int xort_rescale
        int xort_rescale_type
        int update_xor_tracker_every_n_frames
        int correlation_history_max_capacity

        object predict
        object fft_object
        object ifft_object
        object input_buffer_fft
        object output_buffer_fft
        object input_buffer_ifft
        object output_buffer_ifft
        object kalman
        object xort
        object confirm_buf

    def __cinit__(self, 
                kalman: KalmanFilter=None,
                FLOAT_TYPE_t alpha_kalman_rate=1/30, 
                bool skip_frames=False, 
                int max_skipped_frames=1, 
                bool debug=False, 
                FLOAT_TYPE_t alpha_smoothing=0.9, 
                int training_images_count=9, 
                FLOAT_TYPE_t output_sigma_factor=0.05,
                FLOAT_TYPE_t correlation_target=0.5,
                tuple rotation_range=(-3, 3),
                tuple scale_range=(0.95, 1.05),
                list scale_factors=[0.7, 0.85, 1.0],   
                bool is_recovery_enabled=True):
        
        self.is_debugging = debug
        self.current_roi = (0, 0, 0, 0)
        self.current_point = (0, 0)
        self.current_global_point = (0, 0)
        self.predicted_global_point = (0, 0)
        self.last_real_global_point = (0, 0)
        self.template_size = (0, 0)
        self.template_size_base = (0, 0)
        self.original_roi = (0, 0, 0, 0)
        self.high_correlation_template_roi = (0, 0, 0, 0)
        self.last_max_correlation = 0.0
        self.correlation_target = correlation_target
        self.min_correlation_for_update = 0.8
        self.multi_scale_detection_good_correlation = 0.7
        self.training_images_count = training_images_count
        self.epsilon = 1e-6
        self.adaptive_rate = 0.0
        self.correlation_history_max_capacity = 5
        self.correlation_history = []
        self.frame_count = -1

        ## Kalman Filter
        if kalman is None:
            self.kalman = KalmanFilter(self.kalman_rate)
        else:
            self.kalman = kalman
        self.predict = self.kalman.cv2_predict
        self.kalman_rate = alpha_kalman_rate
        
        ## Tracking
        self.is_tracking = False
        self.tracking_lost_max_attempts = 12
        self.tracking_lost_current_attempt = 0
        self.d_max = 15
        self.low_correlation_frame_count = 0
        self.low_correlation_frame_count_max = 5
        self.high_correlation_threshold = 0.9
        self.correlation_deviation = 0
              
        ## Multiscale detection
        self.scale_factors = scale_factors
        self.expanded_scale_factors = list((*scale_factors, 1.15, 1.3))
        self.current_template_scale = 1.0
        self.default_scale_frame_count = 30
        self.skip_multiscale_detection_frames = 3
        self.current_skipped_frames_multiscale_detection = 0
            
        ## Frame skipping
        self.skip_frames = skip_frames
        self.skipped_frame_count = 0
        self.max_skipped_frames = max_skipped_frames
        self.delay_first_frames = 10

        ## Training
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.output_sigma_factor = output_sigma_factor
    
        # Recovery
        self.is_recovery_enabled = is_recovery_enabled
        self.tracker_bank_max_count = 10
        self.tracker_bank = []
        self.frame_count_from_last_recovery = 0
        self.is_tracking_recovered = False
        self.tracker_bank_index = 0
        
        # Point smoothness
        self.smoothed_x = 0
        self.smoothed_y = 0
        self.alpha_smoothing = alpha_smoothing
        
        #PyFFTW
        self.fft_object = None
        self.ifft_object = None
        self.input_buffer_fft = None
        self.output_buffer_fft = None
        self.input_buffer_ifft = None
        self.output_buffer_ifft = None
        self.num_pyfftw_threads = 0
        
        # XOR Tracker
        self.xort = XORTracker()
        self.xort_rescale = 2
        self.xort_rescale_type = cv2.INTER_NEAREST
        self.confirm_buf = deque(maxlen=5)
        self.update_xor_tracker_every_n_frames = 3
        self.search_strategies = [
            (10, lambda im: (int(im.shape[1] * 0.85), int(im.shape[0] * 0.85))),
            (8,  lambda im: (int(im.shape[1] * 0.7),  int(im.shape[0] * 0.7))),
            (6,  lambda im: (im.shape[1] // 2,        int(im.shape[0] * 0.85))),
            (4,  lambda im: (int(im.shape[1] * 0.85), im.shape[0] // 2)),
            (2,  lambda im: (im.shape[1] // 2,        im.shape[0] // 2)),
            (0,  lambda im: (im.shape[1] // 3,        im.shape[0] // 3)),
        ]

        # Debug
        self.debug = {}
        
        
    cpdef bool init(self, np.ndarray[np.uint8_t, ndim=2] im, tuple roi):
        cdef:
            int x, y, w, h
            tuple template_data
            np.ndarray template
            np.ndarray[np.uint8_t, ndim=2] image
            np.ndarray[np.uint8_t, ndim=3] training_images
            np.ndarray[FLOAT_TYPE_t, ndim=2] target, template_processed, result, preprocessed_image, processed_image
            np.ndarray[COMPLEX_TYPE_t, ndim=2] synthetic_target_f, preprocessed_image_f, weight, energy

        x, y, w, h = roi

        self.template_size = (h, w)
        self.template_size_base = self.template_size
        
        template = im[y:y+h, x:x+w]
        self.init_template = template.copy()
        self.high_correlation_template = template.copy()
        self.init_frame = im.copy()
        self.high_correlation_frame = im.copy()
        self.original_roi = roi
        self.high_correlation_template_roi = roi
        self.init_xor_tracker(im, roi)

        self.input_buffer_fft = pyfftw.empty_aligned((h, w), dtype=complex_type_name)
        self.output_buffer_fft = pyfftw.empty_aligned((h, w), dtype=complex_type_name)
        self.input_buffer_ifft = self.output_buffer_fft
        self.output_buffer_ifft = pyfftw.empty_aligned((h, w), dtype=complex_type_name)
        self.fft_object = pyfftw.FFTW(self.input_buffer_fft,
                                      self.output_buffer_fft,
                                      axes=(0, 1),
                                      direction="FFTW_FORWARD",
                                      flags=("FFTW_MEASURE",),
                                      threads=self.num_pyfftw_threads)
        self.ifft_object = pyfftw.FFTW(self.input_buffer_ifft,
                                      self.output_buffer_ifft,
                                      axes=(0, 1),
                                      direction="FFTW_BACKWARD",
                                      flags=("FFTW_MEASURE",),
                                      threads=self.num_pyfftw_threads)

        target = make_synthetic_with_regularization(self.template_size[0], 
                                           self.template_size[1], 
                                           np.array([(self.template_size[0] / 2.0, self.template_size[1] / 2.0)], dtype=FLOAT_TYPE), 
                                           output_sigma_factor=self.output_sigma_factor)
        self.centric_synthetic_target = target.copy()

        synthetic_target_f = self._compute_fft(target)
        self.centric_synthetic_target_f = synthetic_target_f.copy()
        training_images = self._generate_training_images(template)
        weight = np.zeros((h, w), dtype=COMPLEX_TYPE)
        energy = np.zeros((h, w), dtype=COMPLEX_TYPE)
        for i, image in enumerate(training_images):
            processed_image = self._process_template(image)
            preprocessed_image = image_preprocessing(processed_image)
            preprocessed_image_f = self._compute_fft(preprocessed_image)
            weight += synthetic_target_f * preprocessed_image_f.conj()
            energy += preprocessed_image_f * preprocessed_image_f.conj()

        self.current_weight = weight
        self.current_energy = energy
        self.computed_filter = weight / (energy + self.epsilon)
        template_processed = self._process_template(template)
        self.init_template_f = self._compute_fft(image_preprocessing(template_processed))
        self._add_data_to_tracker_bank((template, self.init_template_f, im, roi))
        self.high_correlation_template_f = self.init_template_f.copy()
        result = self._compute_ifft(self.computed_filter * self.init_template_f)
        self.last_max_correlation = min(np.max(result), 1.0)
        print("Init max correlation:", self.last_max_correlation)
        self.correlation_history.append(self.last_max_correlation)
        self.current_point = (self.template_size[0] // 2, self.template_size[1] // 2)
        self.current_global_point = (int(self.current_point[0] + y), int(self.current_point[1] + x))
        self._smooth_current_point()

        self.correlation_deviation = self._calculate_correlation_deviation()

        self.predicted_global_point = self.predict(self.current_global_point[0], self.current_global_point[1])
        self.is_tracking = True     
        return self.is_tracking

    cpdef tuple update(self, np.ndarray[np.uint8_t, ndim=2] im):
        if self.computed_filter is None:
            raise Exception("Init the tracker first!")
        cdef: 
            tuple template_data = self._extract_template(im, self.current_global_point, self.template_size)
            tuple roi = template_data[1]
            int x = roi[0]
            int y = roi[1]
            int w = roi[2]
            int h = roi[3]
            int new_x, new_y, new_h, new_w, dy, dx
            np.ndarray[np.uint8_t, ndim=2] template = template_data[0]
            np.ndarray[FLOAT_TYPE_t, ndim=2] target, template_processed, correlation_map
            np.ndarray[COMPLEX_TYPE_t, ndim=2] target_f, preprocessed_template_f
            tuple new_point, best_roi
            FLOAT_TYPE_t new_template_scale, apce, current_max_correlation
            bool is_xor_tracking_good = False, is_xor_tracking_excellent = False
        
        self.frame_count += 1 

        if self.skip_frames:
            if self.delay_first_frames > 0:
                self.delay_first_frames -= 1
            elif self.skipped_frame_count < self.max_skipped_frames and self.current_global_point != (0, 0):
                self.skipped_frame_count += 1
                self.current_global_point = self.predict(self.current_global_point[0], self.current_global_point[1])
                new_y = self.predicted_global_point[0] - self.template_size[0] // 2
                new_x = self.predicted_global_point[1] - self.template_size[1] // 2
                new_h = self.template_size[0]
                new_w = self.template_size[1]
                    
                if self.is_debugging:
                    self.debug = {
                    "template": template,
                    "filter_result": None,
                    "target": None 
                    }

                self.current_roi = (new_x, new_y, new_w, new_h)
                return self.is_tracking, self.current_roi, self.debug
            else:
                self.skipped_frame_count = 0
        
        if self.frame_count > self.default_scale_frame_count:  
            if self.current_skipped_frames_multiscale_detection < self.skip_multiscale_detection_frames:
                self.current_skipped_frames_multiscale_detection += 1 
                new_point, best_roi, current_max_correlation, new_template_scale, correlation_map, preprocessed_template_f = self._multi_scale_detection(im, self.current_global_point, self.current_template_scale)
            else:
                new_point, best_roi, current_max_correlation, new_template_scale, correlation_map, preprocessed_template_f = self._multi_scale_detection(im, self.current_global_point)
                self.current_skipped_frames_multiscale_detection = 0
        else:
            template_processed = self._process_template(template)
            preprocessed_template_f = self._compute_fft(image_preprocessing(template_processed))
            correlation_map = self._compute_ifft(self.computed_filter * preprocessed_template_f)
            current_max_correlation = np.max(correlation_map)
            current_max_correlation = min(current_max_correlation, 1.0)
            new_point = np.unravel_index(np.argmax(correlation_map), (correlation_map.shape[0], correlation_map.shape[1]))
            best_roi = roi
            new_template_scale = 1.0
        
        self.correlation_history.append(current_max_correlation)

        dy = np.abs(self.current_point[0] - new_point[0])
        dx = np.abs(self.current_point[1] - new_point[1])
        if int(np.ceil(dy)) <= self.d_max and int(np.ceil(dx)) <= self.d_max:
            self.current_point = new_point
            target = make_synthetic_with_regularization(h, w, np.array([(self.current_point[0], self.current_point[1])], dtype=FLOAT_TYPE),
            output_sigma_factor=self.output_sigma_factor)
        else:
            self.current_point = (self.template_size[0] // 2, self.template_size[1] // 2)
            target = self.centric_synthetic_target.copy()
        target_f = self._compute_fft(target)
        
        if current_max_correlation > self.correlation_target:   
            self.current_weight = self.adaptive_rate * (target_f * preprocessed_template_f.conj()) + (1.0 - self.adaptive_rate) * self.current_weight
            self.current_energy = self.adaptive_rate * (preprocessed_template_f * preprocessed_template_f.conj() + self.epsilon) + (1.0 - self.adaptive_rate) * self.current_energy
            self.computed_filter = self.current_weight / (self.current_energy + self.epsilon)

            if current_max_correlation > self.high_correlation_threshold:
                self._add_data_to_tracker_bank((template, preprocessed_template_f, im, best_roi))
        
            correlation_map = self._compute_ifft(self.computed_filter * preprocessed_template_f)
            apce = self._calculate_apce(correlation_map, template)
            self._update_adaptive_learning_rate(apce)

        cdef FLOAT_TYPE_t previous_correlations_avg

        if self.is_tracking:
            correlation_deviation = self.correlation_deviation
        else:
            correlation_deviation = self.correlation_deviation * 1.25
        
        if len(self.correlation_history) > self.correlation_history_max_capacity:
            self.correlation_history.pop(0)

        previous_correlations_avg = np.mean(self.correlation_history[:-1])
            
        self._update_tracking_state(im, current_max_correlation, previous_correlations_avg, correlation_deviation)
            
        if self.is_tracking_recovered or self.is_tracking:
            self.current_global_point  = (y + int(self.current_point[0]), x + int(self.current_point[1]))
            self.current_template_scale = new_template_scale
            self.current_roi = best_roi
        
        if self.is_tracking:

            self.predicted_global_point = self.predict(self.current_global_point[0], self.current_global_point[1])
            self.last_real_global_point = self.current_global_point

            if self.frame_count % self.update_xor_tracker_every_n_frames == 0:
                self.update_xor_tracker(im, x, y)
        elif self.is_recovery_enabled and not self.is_tracking_recovered:
            if self.tracking_lost_current_attempt > self.tracking_lost_max_attempts:

                self.high_correlation_template = self.tracker_bank[self.tracker_bank_index][0]
                self.high_correlation_template_f = self.tracker_bank[self.tracker_bank_index][1]
                self.high_correlation_frame = self.tracker_bank[self.tracker_bank_index][2]
                self.high_correlation_template_roi = self.tracker_bank[self.tracker_bank_index][3]
                
                self.reset_xor_tracker(self.high_correlation_frame, self.high_correlation_template_roi)
                self.reset_filter_with_high_correlation_template()
                self.tracker_bank_index -= 1
                if self.tracker_bank_index < 0:
                    self.tracker_bank_index = len(self.tracker_bank) - 1
                self.tracking_lost_current_attempt = 0

            for threshold, stategy in self.search_strategies:
                if self.tracking_lost_current_attempt >= threshold:
                    x, y = stategy(im)
                    print(float(x) / im.shape[1], float(y) / im.shape[0])
                    x, y, is_xor_tracking_good = self.target_search(im, x, y)
                    break
                
            self.current_roi = (x, y, w, h)
            self.current_global_point = (y + h // 2, x + w // 2)

            new_point, best_roi, current_max_correlation, new_template_scale, correlation_map, preprocessed_template_f = self._multi_scale_detection(im, self.current_global_point, scales=self.expanded_scale_factors)

            if current_max_correlation > self.correlation_target:
                self.correlation_history.append(current_max_correlation)
               
                self.current_template_scale = new_template_scale
                self.adaptive_rate = 0.3
                self.current_roi = roi
                self.current_point = new_point
                self.current_skipped_frames_multiscale_detection = 0
                self.tracking_lost_current_attempt = 0
                self.low_correlation_frame_count = 0
                self.frame_count_from_last_recovery = 0
                self.is_tracking_recovered = True
            else:
                self.tracking_lost_current_attempt += 1            

        self.last_max_correlation = current_max_correlation
        
        if self.is_debugging:   
            self.debug = {
                "template": template,
                "filter_result": correlation_map,
                "target": target,
                "adaptive_rate": self.adaptive_rate,
                "computed_filter": np.abs(np.fft.fftshift(self.computed_filter.copy())),
            } 
           
        return self.is_tracking, self.current_roi, self.debug
    
    
    cpdef dict get_tracker_data(self):
        data = {
            "roi": list(map(int, self.current_roi)),
            "correlation": float(self.last_max_correlation),
            "template_scale": float(self.current_template_scale),
            "learning_rate": float(self.adaptive_rate),
            "correlation_target": float(self.correlation_target)
        }
        return data


    cpdef void reset_filter_with_original_template(self):
        self.current_weight = self.centric_synthetic_target_f * self.init_template_f.conj()
        self.current_energy = self.init_template_f * self.init_template_f.conj()
        self.computed_filter = self.current_weight / (self.current_energy + self.epsilon)


    cpdef void reset_filter_with_high_correlation_template(self):
        self.current_weight = self.centric_synthetic_target_f * self.high_correlation_template_f.conj()
        self.current_energy = self.high_correlation_template_f * self.high_correlation_template_f.conj()
        self.computed_filter = self.current_weight / (self.current_energy + self.epsilon)
    

    cpdef void init_xor_tracker(self, np.ndarray[np.uint8_t, ndim=2] im, tuple roi):
        im = cv2.resize(im, (im.shape[1] // self.xort_rescale, im.shape[0] // self.xort_rescale), self.xort_rescale_type)
        roi = (roi[0] // self.xort_rescale, roi[1] // self.xort_rescale, roi[2] // self.xort_rescale, roi[3] // self.xort_rescale)
        self.xort.init(im, roi)


    cpdef tuple target_search(self, np.ndarray[np.uint8_t, ndim=2] im, int width, int height):
        im = cv2.resize(im, (im.shape[1] // self.xort_rescale, im.shape[0] // self.xort_rescale), self.xort_rescale_type)
        width /= self.xort_rescale
        height /= self.xort_rescale
        x, y, is_good = self.xort.target_search(im, width, height)
        return x * self.xort_rescale, y * self.xort_rescale, is_good


    cpdef void update_xor_tracker(self, np.ndarray[np.uint8_t, ndim=2] im, int x, int y):
        im = cv2.resize(im, (im.shape[1] // self.xort_rescale, im.shape[0] // self.xort_rescale), self.xort_rescale_type)
        x /= self.xort_rescale
        y /= self.xort_rescale
        self.xort.update_shift_frame(im, x, y)
        self.xort.refresh_mask(x, y)
        self.xort.set_xy_pos(x, y)


    cpdef void reset_xor_tracker(self, np.ndarray[np.uint8_t, ndim=2] im, tuple roi):
        im = cv2.resize(im, (im.shape[1] // self.xort_rescale, im.shape[0] // self.xort_rescale), self.xort_rescale_type)
        roi = (roi[0] // self.xort_rescale, roi[1] // self.xort_rescale, roi[2] // self.xort_rescale, roi[3] // self.xort_rescale)
        self.xort.calculate_mask(im, roi)


    cpdef void _update_tracking_state(self, 
                                  np.ndarray[np.uint8_t, ndim=2] im,
                                  FLOAT_TYPE_t current_max_correlation,
                                  FLOAT_TYPE_t previous_correlations_avg,
                                  FLOAT_TYPE_t correlation_deviation): 
        cdef:
            FLOAT_TYPE_t rel_drop_threshold = 0.35
            FLOAT_TYPE_t abs_drop_threshold = max(0.15, correlation_deviation)
            FLOAT_TYPE_t corr_confirm_thresh = self.correlation_target
            int min_confirm_frames = 3
            int max_recovery_frames = 8
            FLOAT_TYPE_t prev_safe
            FLOAT_TYPE_t absolute_drop
            FLOAT_TYPE_t relative_drop

        prev_safe = max(previous_correlations_avg, self.epsilon)
        absolute_drop = prev_safe - current_max_correlation
        relative_drop = absolute_drop / prev_safe if prev_safe > 0 else 1.0

        # --- Состояние: идёт подтверждение кандидата ---
        if self.is_tracking_recovered:
            if current_max_correlation < self.correlation_target * 0.7:
                # кандидат оказался плохим — отменяем
                self.is_tracking_recovered = False
                self.frame_count_from_last_recovery = 0
                self.confirm_buf.clear()
                return

            # копим подтверждения
            if current_max_correlation >= corr_confirm_thresh:
                self.confirm_buf.append(1)
            else:
                self.confirm_buf.append(0)

            # если подтверждений хватает
            if sum(self.confirm_buf) >= min_confirm_frames:
                self.is_tracking = True
                self.is_tracking_recovered = False
                self.frame_count_from_last_recovery = 0
                self.low_correlation_frame_count = 0
                self.confirm_buf.clear()
                return

            # если слишком долго подтверждаем — сбрасываем
            self.frame_count_from_last_recovery += 1
            if self.frame_count_from_last_recovery > max_recovery_frames:
                self.is_tracking_recovered = False
                self.frame_count_from_last_recovery = 0
                self.confirm_buf.clear()
            return

        # --- Состояние: нормальный трекинг ---
        if self.is_tracking:
            if current_max_correlation < self.correlation_target * 0.7:
                self.low_correlation_frame_count += 1
            else:
                self.low_correlation_frame_count = max(0, self.low_correlation_frame_count - 1)

            # проверка на потерю
            if (self.low_correlation_frame_count > self.low_correlation_frame_count_max or
                ((absolute_drop > abs_drop_threshold) and 
                (relative_drop > rel_drop_threshold) and 
                current_max_correlation < self.correlation_target)):

                # пытаемся найти заново
                new_point, best_roi, current_max_correlation, new_template_scale, correlation_map, preprocessed_template_f = self._multi_scale_detection(
                    im, self.predicted_global_point, scales=self.expanded_scale_factors)

                if current_max_correlation >= self.correlation_target:
                    print("Tracker is unstable. Using predicted point")
                    self.current_point = new_point
                    self.current_global_point = self.predicted_global_point
                    return

                # нашли кандидата — включаем подтверждение
                self.is_tracking = False
                self.is_tracking_recovered = True
                self.frame_count_from_last_recovery = 0
                self.low_correlation_frame_count = 0
                self.confirm_buf.clear()
            return

        # --- Состояние: полностью потерян (ищем кандидата) ---
        if not self.is_tracking and not self.is_tracking_recovered:
            new_point, best_roi, current_max_correlation, new_template_scale, correlation_map, preprocessed_template_f = self._multi_scale_detection(
                im, self.predicted_global_point, scales=self.expanded_scale_factors)

            if current_max_correlation >= self.correlation_target:
                # нашли кандидата
                self.is_tracking_recovered = True
                self.frame_count_from_last_recovery = 0
                self.confirm_buf.clear()


    cpdef FLOAT_TYPE_t _calculate_correlation_deviation(self):
        cdef:
            FLOAT_TYPE_t template_area_ratio
            FLOAT_TYPE_t frame_diagonal
            FLOAT_TYPE_t template_diagonal
            FLOAT_TYPE_t base_deviation = 0.15
            FLOAT_TYPE_t max_deviation = 0.3
            FLOAT_TYPE_t min_deviation = 0.1
            FLOAT_TYPE_t log_scale_factor = 2
            
        # Отношение площади шаблона к площади кадра
        template_area_ratio = (self.template_size[0] * self.template_size[1]) / \
                            (self.init_frame.shape[0] * self.init_frame.shape[1])
        
        # Отношение диагоналей
        frame_diagonal = np.sqrt(self.init_frame.shape[0]**2 + self.init_frame.shape[1]**2)
        template_diagonal = np.sqrt(self.template_size[0]**2 + self.template_size[1]**2)
        diagonal_ratio = template_diagonal / frame_diagonal
        
        # Итоговое отклонение
        deviation = base_deviation * (1.0 + log_scale_factor * np.log1p(template_area_ratio + diagonal_ratio))
        print(deviation)
        return np.clip(deviation, min_deviation, max_deviation)

        
    cpdef np.ndarray[COMPLEX_TYPE_t, ndim=2] _compute_fft(self, np.ndarray[FLOAT_TYPE_t, ndim=2] image):
        cdef np.ndarray[COMPLEX_TYPE_t, ndim=2] result
        self.input_buffer_fft.real[:] = image
        self.input_buffer_fft.imag[:] = 0.0
        self.fft_object()
        return self.output_buffer_fft.copy()
        
    
    cpdef np.ndarray[FLOAT_TYPE_t, ndim=2] _compute_ifft(self, np.ndarray[COMPLEX_TYPE_t, ndim=2] fft_spectrum):
        cdef np.ndarray[FLOAT_TYPE_t, ndim=2] scaled_real_result
        self.input_buffer_ifft[:] = fft_spectrum
        scaled_real_result = self.ifft_object().real
        return scaled_real_result.copy()
    

    cpdef FLOAT_TYPE_t _calculate_apce(self, np.ndarray[FLOAT_TYPE_t, ndim=2] correlation_map, np.ndarray[np.uint8_t, ndim=2] template):
        cdef: 
            int h, w
            FLOAT_TYPE_t max_correlation, min_correlation
        h, w = template.shape[0], template.shape[1]
        max_correlation = np.max(correlation_map)
        min_correlation = np.min(correlation_map)
        return (max_correlation - min_correlation)**2 / (np.sum((correlation_map - min_correlation)**2) / (h * w))
        
        
    cpdef void _update_adaptive_learning_rate(self, FLOAT_TYPE_t apce):
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
    
    
    cpdef void _smooth_current_point(self):        
        if self.smoothed_x == 0:
            self.smoothed_x = float(self.current_point[1])
            self.smoothed_y = float(self.current_point[0])
        else:
            self.smoothed_x = (1.0 - self.alpha_smoothing) * self.smoothed_x + self.alpha_smoothing * float(self.current_point[1])
            self.smoothed_y = (1.0 - self.alpha_smoothing) * self.smoothed_y + self.alpha_smoothing * float(self.current_point[0])

        self.current_point = (self.smoothed_y, self.smoothed_x)


    cpdef tuple _multi_scale_detection(self, np.ndarray image, tuple center, FLOAT_TYPE_t current_scale = 0, list scales = None):
        cdef:
            FLOAT_TYPE_t best_correlation = -1.0
            FLOAT_TYPE_t max_correlation, scale 
            FLOAT_TYPE_t best_scale = 1.0
        
            np.ndarray[FLOAT_TYPE_t, ndim=2] best_correlation_map, correlation_map, template_processed
            np.ndarray[np.uint8_t, ndim=2] template
            np.ndarray[COMPLEX_TYPE_t, ndim=2] preprocessed_template_f, best_template
            tuple best_roi
            tuple best_point = (0, 0)
            tuple roi
            list scale_factors
            int h_base, w_base
            int h_scaled, w_scaled
            
        h_base, w_base = self.template_size_base
        
        if current_scale != 0:
            scale_factors = [current_scale]
        else:
            if scales is not None:
                scale_factors = scales
            elif self.last_max_correlation > self.multi_scale_detection_good_correlation:
                scale_factors = [self.current_template_scale] 
            else:
                scale_factors = self.scale_factors

        for scale in scale_factors:
            h_scaled = int(h_base * scale)
            w_scaled = int(w_base * scale)
            
            template, roi = self._extract_template(image, center, (h_scaled, w_scaled))
            
            if scale != 1.0:
                template = cv2.resize(template, (w_base, h_base), interpolation=cv2.INTER_LINEAR)

            if np.std(template) < self.epsilon:
                continue

            template_processed = self._process_template(template)       
            preprocessed_template_f = self._compute_fft(image_preprocessing(template_processed)) 
            correlation_map = self._compute_ifft(self.computed_filter * preprocessed_template_f)
            max_correlation = np.max(correlation_map)
            
            if max_correlation > best_correlation:     
                best_roi = roi
                best_scale = scale
                best_template = preprocessed_template_f
                best_correlation = max_correlation
                best_correlation_map = correlation_map
        best_correlation = min(best_correlation, 1.0)    
        best_point = np.unravel_index(np.argmax(best_correlation_map), (best_correlation_map.shape[0], best_correlation_map.shape[1]))
        best_point = self.subpixel_peak(best_correlation_map, best_point)
        return best_point, best_roi, best_correlation, best_scale, best_correlation_map, best_template


    cdef tuple subpixel_peak(self, np.ndarray[FLOAT_TYPE_t, ndim=2] qmap, tuple yx):
        cdef:
            int h, w, x, y
            FLOAT_TYPE_t f1, f2, f3, denom, dx, dy
        y, x = yx
        h, w = qmap.shape[0], qmap.shape[1]

        f1, f2, f3 = qmap[y, x-1], qmap[y, x], qmap[y, x+1]
        denom = 2 * (f1 - 2*f2 + f3)
        dx = 0 if denom == 0 else (f1 - f3) / denom

        f1, f2, f3 = qmap[y-1,x], qmap[y, x], qmap[y+1,x]
        denom = 2 * (f1 - 2*f2 + f3)
        dy = 0 if denom == 0 else (f1 - f3) / denom
        return y + dy, x + dx


    cpdef void _add_data_to_tracker_bank(self, tuple data):
        cdef:
            np.ndarray[np.uint8_t, ndim=2] template = data[0]
            int h = template.shape[0]
            int w = template.shape[1]
            int area = h * w
            int diff_count
            float rel_thresh = 0.05  # допустимая доля отличий
            int abs_tol = 30         # порог по интенсивности для absdiff

        for td in self.tracker_bank:
            existing = td[0]

            d = cv2.absdiff(existing, template)
            _, mask = cv2.threshold(d, abs_tol, 255, cv2.THRESH_BINARY)
            diff_count = int(cv2.countNonZero(mask))

            if diff_count <= int(rel_thresh * area):
                return

        self.tracker_bank.append(data)
        if len(self.tracker_bank) > self.tracker_bank_max_count:
            self.tracker_bank.pop(1)
        self.tracker_bank_index = len(self.tracker_bank) - 1
    
    
    cpdef tuple _extract_template(self, np.ndarray[np.uint8_t, ndim=2] image, 
                                tuple center, tuple size):
        cdef:
            int h, w, cy, cx, x1, y1
            np.ndarray[np.uint8_t, ndim=2] template
            
        h, w = size
        cy, cx = center
        x1 = np.clip(cx - w // 2, 0, image.shape[1] - w)
        y1 = np.clip(cy - h // 2, 0, image.shape[0] - h)
        template = image[y1:y1 + h, x1:x1 + w]
        return template, (x1, y1, w, h)

    cpdef np.ndarray[np.uint8_t, ndim=3] _generate_training_images(self, np.ndarray[np.uint8_t, ndim=2] template):
        cdef:
            int h, w, remaining_count, rotation_count, scale_count
            tuple center
            np.ndarray training_images
            FLOAT_TYPE_t angle, scale
            int new_w, new_h, start_x, start_y, pad_x, pad_y
            np.ndarray rotation_matrix, rotated, scaled, cropped, padded
            
        if self.training_images_count <= 1:
            return np.array([template])
            
        training_images = np.array([template], dtype=np.uint8)
        h, w = template.shape[0], template.shape[1]
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
                np.append(training_images, rotated)

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
                    np.append(training_images, cropped)       
                else:
                    pad_x = (w - new_w) // 2
                    pad_y = (h - new_h) // 2
                    padded = cv2.copyMakeBorder(
                        scaled, pad_y, h - new_h - pad_y, pad_x, w - new_w - pad_x,
                        cv2.BORDER_ISOLATED
                    )
                    np.append(training_images, padded)
        
        return training_images
    
    
    
    cpdef np.ndarray[FLOAT_TYPE_t, ndim=2] _create_blurry_model(self, 
                                                               np.ndarray template,
                                                               FLOAT_TYPE_t gaussian_sigma_factor=0.0475):
        cdef:
            np.ndarray[FLOAT_TYPE_t, ndim=2] template_float, background_model
            FLOAT_TYPE_t gaussian_sigma
            
        template_float = template.astype(FLOAT_TYPE)
        gaussian_sigma = np.max((template_float.shape[0], template_float.shape[1])) * gaussian_sigma_factor
        
        if gaussian_sigma < 1.0:
            gaussian_sigma = 1
        
        background_model = cv2.GaussianBlur(template_float, (0, 0), sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)
        return background_model
    
    cpdef np.ndarray[FLOAT_TYPE_t, ndim=2] _apply_model_subtraction(self, 
                                                                   np.ndarray image,
                                                                   np.ndarray model):
        cdef:
            np.ndarray[FLOAT_TYPE_t, ndim=2] foreground
            FLOAT_TYPE_t fg_min, fg_max
            
        if model is None:
            return image
        
        foreground = image.astype(FLOAT_TYPE) - model.astype(FLOAT_TYPE)
        
        fg_min = np.min(foreground)
        fg_max = np.max(foreground)
        if fg_max - fg_min > 0:
            foreground = ((foreground - fg_min) / (fg_max - fg_min)) * 255.0

        return foreground.astype(FLOAT_TYPE)
    
    cpdef np.ndarray[FLOAT_TYPE_t, ndim=2] _process_template(self, 
                                                            np.ndarray[np.uint8_t, ndim=2] template,
                                                            FLOAT_TYPE_t gaussian_sigma_factor=0.0475):
        cdef:
            np.ndarray[FLOAT_TYPE_t, ndim=2] template_float
            FLOAT_TYPE_t gaussian_sigma
            np.ndarray background_model
            
        template_float = template.astype(FLOAT_TYPE)
        gaussian_sigma = max(1.0, np.max((template_float.shape[0], template_float.shape[1])) * gaussian_sigma_factor)
        
        background_model = cv2.GaussianBlur(template_float, (0, 0), 
                                        sigmaX=gaussian_sigma, 
                                        sigmaY=gaussian_sigma)
        
        cv2.subtract(template_float, background_model, template_float)
        cv2.normalize(template_float, template_float, 0, 255, cv2.NORM_MINMAX)
        
        return template_float
