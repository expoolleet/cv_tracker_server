import os
os.environ["PYOPENGL_PLATFORM"]="egl"
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np


def create_shader_module(shader_source_code, shader_type):
    return compileShader(shader_source_code, shader_type)


_yuv2rgb_vertex_code = """
    precision mediump float;

    attribute vec2 aPosition;
    attribute vec2 aTexCoord;

    varying vec2 uv;

    void main() {
        uv = aTexCoord;
        gl_Position = vec4(aPosition, 0.0, 1.0);
    }
"""

_yuv2rgb_fragment_code = """
    precision mediump float;

    uniform sampler2D yTex;
    uniform sampler2D uTex;
    uniform sampler2D vTex;

    uniform vec4 roi;

    varying vec2 uv;

    vec3 yuv2rgb();
    vec3 draw_rectangle(vec3 color, float x, float y, float w, float h);
    vec3 draw_crosshair(vec3 color, float size, float thickness);
    vec3 rectangle_color = vec3(1.0, 1.0, 1.0);
    vec3 crosshair_color = vec3(1.0, 1.0, 1.0);
    float crosshair_size = 0.015;
    float crosshair_thickness = 0.003;

    float rectangle_vertical_thickness = 0.003;
    float rectangle_horizontal_thickness = 0.0036; //rectangle_vertical_thickness * 1.2

    void main() {

        vec3 rgb = yuv2rgb();

        vec3 color = draw_rectangle(rgb, roi.x, roi.y, roi.z, roi.w);
        color = draw_crosshair(color, crosshair_size, crosshair_thickness);
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
        float u = texture2D(uTex, pos).r - 0.5;
        float v = texture2D(vTex, pos).r - 0.5; 

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

class OpenGLRenderer:
    def __init__(self, buffer_size=(640, 480), window_x_offset=0, window_y_offset=0):
        self.pos_attrib = 0
        self.color_attrib = 1
        self.textures = []
        self.buffers = []
        
        glfw.init()
        self.monitor = glfw.get_primary_monitor()
        self.mode = glfw.get_video_mode(self.monitor)
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
        
        # setup constants
        global SCREEN_WIDTH, SCREEN_HEIGHT
        SCREEN_WIDTH = buffer_size[0]
        SCREEN_HEIGHT = buffer_size[1]
        print(f"Display parameters:\nWidth: {SCREEN_WIDTH}\nHeight: {SCREEN_HEIGHT}\nRefresh rate: {self.mode.refresh_rate}")
        
        self.window = glfw.create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Preview", None, None)
        glfw.set_window_pos(self.window, window_x_offset, window_y_offset)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        glfw.make_context_current(self.window)
        
        self.create_canvas()
    
    
    def init_yuv2rgb_shader(self) -> None:
        self.create_yuv2rgb_shader_program()
        self.init_stream_texture_yuv2rgb()
    
    
    def create_yuv2rgb_shader_program(self) -> None:
        vertex_module = create_shader_module(_yuv2rgb_vertex_code, GL_VERTEX_SHADER)
        fragment_module = create_shader_module(_yuv2rgb_fragment_code, GL_FRAGMENT_SHADER)
        
        self.yuv2rgb_shader = compileProgram(vertex_module, fragment_module)
        
        glDeleteShader(vertex_module)
        glDeleteShader(fragment_module)
        
        
    def init_stream_texture_yuv2rgb(self) -> None:
        glUseProgram(self.yuv2rgb_shader)
        glUniform1i(glGetUniformLocation(self.yuv2rgb_shader, "yTex"), 0)
        glUniform1i(glGetUniformLocation(self.yuv2rgb_shader, "uTex"), 1)
        glUniform1i(glGetUniformLocation(self.yuv2rgb_shader, "vTex"), 2)
        
        self.y_tex, self.u_tex, self.v_tex = glGenTextures(3)
        self.textures.append(self.y_tex)
        self.textures.append(self.u_tex)
        self.textures.append(self.v_tex)
        glBindTexture(GL_TEXTURE_2D, self.y_tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, None)
        
        glBindTexture(GL_TEXTURE_2D, self.u_tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, None)
        
        glBindTexture(GL_TEXTURE_2D, self.v_tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, None)
        
        glBindTexture(GL_TEXTURE_2D, 0)
        
    
    def diplay_yuv_frame(self, frame_yuv: np.ndarray, roi: np.ndarray) -> None:
        glUseProgram(self.yuv2rgb_shader)
        w, h = (SCREEN_WIDTH, SCREEN_HEIGHT)
        y_size = w * h
        u_size = w * h // 4
        v_size = u_size
        
        y = np.frombuffer(frame_yuv, dtype=np.uint8, count=y_size)
        u = np.frombuffer(frame_yuv, dtype=np.uint8, count=u_size, offset=y_size)
        v = np.frombuffer(frame_yuv, dtype=np.uint8, count=v_size, offset=y_size+u_size)
        
        glUniform4fv(glGetUniformLocation(self.yuv2rgb_shader, "roi"), 1, roi)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.y_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_LUMINANCE, GL_UNSIGNED_BYTE, y)
        
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.u_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w // 2, h // 2, GL_LUMINANCE, GL_UNSIGNED_BYTE, u)
        
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self.v_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w // 2, h // 2, GL_LUMINANCE, GL_UNSIGNED_BYTE, v)
        
        self.draw_canvas()       
        glfw.swap_buffers(self.window)
      
        
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
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, None)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glDisableVertexAttribArray(position_attribute_index)
        glDisableVertexAttribArray(texture_coordinates_attribute_index)
        
        
    def quit(self) -> None:
        glDeleteProgram(self.yuv2rgb_shader)
        glDeleteBuffers(len(self.buffers), self.buffers)
        glDeleteTextures(len(self.textures), self.textures)
        glfw.destroy_window(self.window)
        glfw.terminate()
        
    