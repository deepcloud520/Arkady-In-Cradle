import moderngl
import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import QTimer


class GLWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)
        
        self.ctx = None
        self.program = None
        self.vao = None
        self.time = 0.0
    
    def initializeGL(self):
        try:
            self.ctx = moderngl.create_context()
            print(f"OpenGL version: {self.ctx.version_code}")
            vertex_shader = """
        #version 330 core
        in vec2 in_vert;
        in vec3 in_color;
        out vec3 v_color;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            v_color = in_color;
        }
        """

            fragment_shader = """
        #version 330 core
        in vec3 v_color;
        out vec4 frag_color;
        uniform float time;
        void main() {
            // 大幅动态变化，便于验证
            frag_color = vec4(v_color * (0.5 + 0.5 * sin(time * 3.0)), 1.0);
        }"""
            
            self.program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            
            vertices = np.array([
                -0.5, -1.0, 0.1, 0.2, 0.4,
                 1.0, -1.0, 0.2, 0.3, 0.5,
                 0.0,  1.0, 0.3, 0.4, 0.6,
            ], dtype=np.float32)
            
            vbo = self.ctx.buffer(vertices)
            self.vao = self.ctx.vertex_array(
                self.program,
                [(vbo, '2f 3f', 'in_vert', 'in_color')]
            )
            
            self.ctx.clear_color = (0.05, 0.05, 0.1, 1.0)
            fbo = self.ctx.detect_framebuffer()
            if fbo:
                self.ctx.fbo = fbo
            else:
                self.ctx.fbo = None
        except Exception as e:
            print(f"initializeGL error: {e}")
            import traceback
            traceback.print_exc()
    
    def paintGL(self):
        if self.ctx is None or self.vao is None:
            return
        
        try:
            

            self.ctx.clear()
            self.makeCurrent()
            #self.ctx.clear()
            self.time += 0.016
            
            self.program['time'] = self.time
            self.vao.render(moderngl.TRIANGLES)
        except Exception as e:
            print(f"paintGL error: {e}")
    
    def resizeGL(self, width, height):
        if self.ctx is not None:
            self.ctx.viewport = (0, 0, width, height)