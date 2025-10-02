import os
os.environ["PYOPENGL_PLATFORM"]="egl"
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np
import threading


def create_shader_module(shader_source_code, shader_type):
    return compileShader(shader_source_code, shader_type)

# shader_binaries_dir =

clear_color = (0, 0, 0, 1)

_vertex_code = """
    precision mediump float;

    attribute vec2 aPosition;
    attribute vec2 aTexCoord;

    uniform mat4 model_view_projection;

    varying vec2 uv;

    void main() {
        uv = aTexCoord;
        gl_Position = model_view_projection * vec4(aPosition, 0.0, 1.0);
    }
"""

_texture_fragment_code = """
    precision mediump float;
    
    uniform sampler2D tex;
    
    varying vec2 uv;
    
    void main() {
        vec2 flipped_uv = vec2(uv.x, 1. - uv.y);
        gl_FragColor = texture2D(tex, flipped_uv);
    }

"""

_yuv2rgb_fragment_code = """
    precision mediump float;

    uniform sampler2D yTex;
    uniform sampler2D uvTex;
    uniform int drawing_crosshair;
    uniform vec4 roi;
    uniform vec3 rectangle_color;
    uniform vec3 crosshair_color;

    varying vec2 uv;

    vec3 yuv2rgb();
    vec3 draw_rectangle(vec3 color, float x, float y, float w, float h);
    vec3 draw_crosshair(vec3 color, float size, float thickness);
    float crosshair_size = 0.012;
    float crosshair_thickness = 0.0025;

    float rectangle_vertical_thickness = 0.0025;
    float rectangle_horizontal_thickness = 0.003; //rectangle_vertical_thickness * 1.2
    
    void main() {

        vec3 rgb = yuv2rgb();

        vec3 color = draw_rectangle(rgb, roi.x, roi.y, roi.z, roi.w);
        if (drawing_crosshair == 1) {
            color = draw_crosshair(color, crosshair_size, crosshair_thickness);
        }
        
        gl_FragColor = vec4(color, 1.0);
    }

    vec3 draw_crosshair(vec3 color, float size, float thickness) {   
        vec2 center = vec2(0.5, 0.5);
        vec2 dist = abs(uv - center);
        float hor_mask = step(dist.y, thickness) * step(dist.x, size);
        float ver_mask = step(dist.x, thickness) * step(dist.y, size);
        float crosshair_mask = max(hor_mask, ver_mask);
        return mix(color, crosshair_color, crosshair_mask);
    }

    vec3 draw_rectangle(vec3 color, float x, float y, float w, float h) {

        y = 1. - y;

        float x_in_range_outside = 1. - step(x - rectangle_horizontal_thickness, uv.x) + step(x + w + rectangle_horizontal_thickness, uv.x);
        float y_in_range_outside = 1. - step(y - h - rectangle_vertical_thickness, uv.y) + step(y + rectangle_vertical_thickness, uv.y) ;

        float x_in_range_inside = step(x + rectangle_horizontal_thickness, uv.x) - step(x + w - rectangle_horizontal_thickness, uv.x);
        float y_in_range_inside = step(y - h + rectangle_vertical_thickness, uv.y) - step(y - rectangle_vertical_thickness, uv.y);
        float in_inside = max(max(x_in_range_outside, y_in_range_outside), x_in_range_inside * y_in_range_inside);

        return color + rectangle_color * (1. - in_inside);
    }

    vec3 yuv2rgb() {
        vec2 pos = uv;
        pos.y = 1. - pos.y;
        float y = texture2D(yTex, pos).r;
        vec2 uv_plane = texture2D(uvTex, pos).ra;
        float u = uv_plane.x - 0.5;
        float v = uv_plane.y - 0.5;

        float r = y + 1.5748 * v;
        float g = y - 0.187324 * u - 0.468124 * v;
        float b = y + 1.8556 * u;
        return vec3(r, g, b);
    }
"""

_canvas_vertex_points = np.array([ # xy, uv
    -1.0, -1.0, 0.0, 0.0,
     1.0, -1.0, 1.0, 0.0,
     1.0,  1.0, 1.0, 1.0,
    -1.0,  1.0, 0.0, 1.0
], dtype=np.float32)

_canvas_vertex_indeces = np.array([0, 1, 2, 2, 3, 0], dtype=np.ubyte)

class ProjectionViewModel:
    KEEP_RATIO = "keep_ratio"
    FREE_RATIO = "free_ratio"

class OpenGLRenderer:
    def __init__(self, buffer_size=(640, 480), window_x_offset=0, window_y_offset=0, vsync=True):
        self.pos_attrib = 0
        self.color_attrib = 1
        self.textures = []
        self.buffers = []

        self.context_lock = threading.Lock()

        glfw.init()
        self.monitor = glfw.get_primary_monitor()
        self.mode = glfw.get_video_mode(self.monitor)
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)

        # setup constants
        self.buffer_width = buffer_size[0]
        self.buffer_height = buffer_size[1]
        self.screen_width = self.mode.size.width
        self.screen_height = self.mode.size.height
        print(f"Display parameters:\nWidth: {self.screen_width}\nHeight: {self.screen_height}\nRefresh rate: {self.mode.refresh_rate}")

        self.window = glfw.create_window(self.screen_width, self.screen_height, "Preview", None, None)
        glfw.set_window_pos(self.window, window_x_offset, window_y_offset)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        glfw.make_context_current(self.window)
        glfw.swap_interval(int(vsync))

        self.create_canvas()     
        glClearColor(*clear_color)

    def init_yuv2rgb_shader(self) -> None:
        self.create_yuv2rgb_shader_program()
        self.init_stream_texture_yuv2rgb()

    def init_texture_shader(self) -> None:
        self.create_texture_shader()
        self.init_stream_texture()

    def create_texture_shader(self) -> None:
        vertex_module = create_shader_module(_vertex_code, GL_VERTEX_SHADER)
        fragment_module = create_shader_module(_texture_fragment_code, GL_FRAGMENT_SHADER)
        self.texture_shader = compileProgram(vertex_module, fragment_module)
        glDeleteShader(vertex_module)
        glDeleteShader(fragment_module)

    def create_yuv2rgb_shader_program(self) -> None:
        vertex_module = create_shader_module(_vertex_code, GL_VERTEX_SHADER)
        fragment_module = create_shader_module(_yuv2rgb_fragment_code, GL_FRAGMENT_SHADER)
        self.yuv2rgb_shader = compileProgram(vertex_module, fragment_module)    
        glDeleteShader(vertex_module)
        glDeleteShader(fragment_module)


    def get_keep_ratio_scale_matrix(self) -> np.ndarray:
        screen_aspect = self.screen_width / self.screen_height
        buffer_aspect = self.buffer_width / self.buffer_height

        if screen_aspect > buffer_aspect:
            scale_y = 1.0
            scale_x = buffer_aspect / screen_aspect
        else:
            scale_x = 1.0
            scale_y = screen_aspect / buffer_aspect

        return np.array([
            [scale_x, 0.0, 0.0, 0.0],
            [0.0, scale_y, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

    def get_free_ratio_scale_matrix(self) -> np.ndarray:
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

    def init_stream_texture_yuv2rgb(self) -> None:
        glUseProgram(self.yuv2rgb_shader)
        glUniform1i(glGetUniformLocation(self.yuv2rgb_shader, "yTex"), 0)
        glUniform1i(glGetUniformLocation(self.yuv2rgb_shader, "uvTex"), 1)
        glUniformMatrix4fv(glGetUniformLocation(self.yuv2rgb_shader, "model_view_projection"), 1, GL_FALSE, self.get_keep_ratio_scale_matrix())
        self.y_tex, self.uv_tex = glGenTextures(2)
        self.textures.append(self.y_tex)
        self.textures.append(self.uv_tex)
        glBindTexture(GL_TEXTURE_2D, self.y_tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, self.buffer_width, self.buffer_height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, self.uv_tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE_ALPHA, self.buffer_width // 2, self.buffer_height // 2, 0, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

    def init_stream_texture(self):
        glUseProgram(self.texture_shader)
        glUniform1i(glGetUniformLocation(self.texture_shader, "tex"), 0)
        glUniformMatrix4fv(glGetUniformLocation(self.texture_shader, "model_view_projection"), 1, GL_FALSE, self.get_keep_ratio_scale_matrix())
        self.rgb_tex = glGenTextures(1)
        self.textures.append(self.rgb_tex)
        glBindTexture(GL_TEXTURE_2D, self.rgb_tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.buffer_width, self.buffer_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

    def change_view_projection_model(self, model: str) -> None:
        if model == ProjectionViewModel.KEEP_RATIO:
            glUseProgram(self.yuv2rgb_shader)
            glUniformMatrix4fv(glGetUniformLocation(self.yuv2rgb_shader, "model_view_projection"), 1, GL_FALSE, self.get_keep_ratio_scale_matrix())
            glUseProgram(self.texture_shader)
            glUniformMatrix4fv(glGetUniformLocation(self.texture_shader, "model_view_projection"), 1, GL_FALSE, self.get_keep_ratio_scale_matrix())
        elif model == ProjectionViewModel.FREE_RATIO:
            glUseProgram(self.yuv2rgb_shader)
            glUniformMatrix4fv(glGetUniformLocation(self.yuv2rgb_shader, "model_view_projection"), 1, GL_FALSE, self.get_free_ratio_scale_matrix())
            glUseProgram(self.texture_shader)
            glUniformMatrix4fv(glGetUniformLocation(self.texture_shader, "model_view_projection"), 1, GL_FALSE, self.get_free_ratio_scale_matrix())

    def set_yuv_frame_drawings(self, roi: np.ndarray = None, drawing_crosshair=True, color=np.array([1, 1, 1], dtype=np.float32)) -> None:
        glUseProgram(self.yuv2rgb_shader)
        if roi is None:
            roi = np.array([0, 0, 0, 0], dtype=np.float32)
        glUniform4fv(glGetUniformLocation(self.yuv2rgb_shader, "roi"), 1, roi)
        glUniform3fv(glGetUniformLocation(self.yuv2rgb_shader, "rectangle_color"), 1, color)
        glUniform3fv(glGetUniformLocation(self.yuv2rgb_shader, "crosshair_color"), 1, color)
        glUniform1i(glGetUniformLocation(self.yuv2rgb_shader, "drawing_crosshair"), int(drawing_crosshair))

    def diplay_yuv_frame(self, frame_yuv: np.ndarray) -> None:
        glUseProgram(self.yuv2rgb_shader)
        glClear(GL_COLOR_BUFFER_BIT)

        w, h = (self.buffer_width, self.buffer_height)
        y_size = w * h
        u_size = w * h // 4
        v_size = u_size
        y_plane = np.frombuffer(frame_yuv, dtype=np.uint8, count=y_size)
        u_plane = np.frombuffer(frame_yuv, dtype=np.uint8, count=u_size, offset=y_size).reshape((h//2, w//2))
        v_plane = np.frombuffer(frame_yuv, dtype=np.uint8, count=v_size, offset=y_size+u_size).reshape((h//2, w//2))
        uv_plane = np.dstack((u_plane, v_plane))

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.y_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_LUMINANCE, GL_UNSIGNED_BYTE, y_plane)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.uv_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w // 2, h // 2, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, uv_plane)        

        err = glGetError()
        if err != GL_NO_ERROR:
            print("OpenGL error:", err)
        self.draw_canvas()       
        glfw.swap_buffers(self.window)

    def display_rgb_frame(self, frame: np.ndarray) -> None:
        glUseProgram(self.texture_shader)
        glClear(GL_COLOR_BUFFER_BIT)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.rgb_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.buffer_width, self.buffer_height, GL_RGBA, GL_UNSIGNED_BYTE, frame)
        self.draw_canvas()
        glfw.swap_buffers(self.window)

    def read_buffer(self, buffer: np.ndarray) -> None:
        glReadPixels(0, 0, self.screen_width, self.screen_height, GL_RGBA, GL_UNSIGNED_BYTE, buffer)

    def create_canvas(self) -> None:      
        self.canvas_vertex_buffer, self.canvas_element_buffer = glGenBuffers(2)
        self.buffers.append(self.canvas_vertex_buffer)
        self.buffers.append(self.canvas_element_buffer)
        glBindBuffer(GL_ARRAY_BUFFER, self.canvas_vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, _canvas_vertex_points.nbytes, _canvas_vertex_points, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.canvas_element_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, _canvas_vertex_indeces.nbytes, _canvas_vertex_indeces, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw_canvas(self) -> None:
        glBindBuffer(GL_ARRAY_BUFFER, self.canvas_vertex_buffer)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.canvas_element_buffer)

        # xy attribute
        offset = 0
        position_attribute_index = 0
        stride = 16 # 4 * sizeof(float32)
        size = 2
        glVertexAttribPointer(position_attribute_index, size, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(position_attribute_index)

        # uv attribute
        offset = 8
        texture_coordinates_attribute_index = 1
        glVertexAttribPointer(texture_coordinates_attribute_index, size, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(texture_coordinates_attribute_index)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, None) # drawing canvas (fullscreen quad)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glDisableVertexAttribArray(position_attribute_index)
        glDisableVertexAttribArray(texture_coordinates_attribute_index)

    def close(self) -> None:
        if self.yuv2rgb_shader:
            glDeleteProgram(self.yuv2rgb_shader)
        if self.texture_shader:
            glDeleteProgram(self.texture_shader)
        glDeleteBuffers(len(self.buffers), self.buffers)
        glDeleteTextures(len(self.textures), self.textures)
        glfw.destroy_window(self.window)
        glfw.terminate()
