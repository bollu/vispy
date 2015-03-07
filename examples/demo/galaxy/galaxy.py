#!/usr/bin/env python
import numpy as np

from vispy.util.transforms import perspective
from  vispy.util import transforms
from vispy import gloo
from vispy import app
from vispy import io

from galaxy_specrend import *
from galaxy_simulation import Galaxy

VERT_SHADER = """
#version 120
uniform mat4  u_model;
uniform mat4  u_view;
uniform mat4  u_projection;
//uniform sampler2D u_colormap;
uniform sampler1D u_colormap;

attribute float a_size;
attribute float a_type;
attribute vec2  a_position;
attribute float a_brightness;
attribute float a_color_index;

varying vec3 v_color;
void main (void)
{   
    gl_Position = u_projection * u_view * u_model * vec4(a_position,0.0,1.0);
    
    vec3 texture_sample = texture1D(u_colormap, a_color_index).rgb;
    v_color = texture_sample * a_brightness;
    
    if (a_size > 2.0)
    {
        gl_PointSize = a_size;
    } else {
        gl_PointSize = 0.0;
    }
    
    if (a_type == 2)
        v_color *= vec3(2,1,1);
    else if (a_type == 3)
        v_color = vec3(.9);
}
"""

FRAG_SHADER = """
#version 120
uniform sampler2D u_texture;
varying vec3 v_color;
void main()
{
    vec3 star_color = v_color;
    float star_tex_white = texture2D(u_texture, gl_PointCoord).r;
    gl_FragColor = vec4(star_color * star_tex_white, 0.8);
}
"""

galaxy = Galaxy(35000)
galaxy.reset(13000, 4000, 0.0004, 0.90, 0.90, 0.5, 200, 300)
t0, t1 = 2000.0, 10000.0
n = 156
dt = (t1 - t0) / n

colors = np.zeros((n, 2), dtype=(np.float32, 3))
for i in range(n):
    temperature = t0 + i * dt
    x, y, z = spectrum_to_xyz(bb_spectrum, temperature)
    r, g, b = xyz_to_rgb(SMPTEsystem, x, y, z)
    r = min((max(r, 0), 1))
    g = min((max(g, 0), 1))
    b = min((max(b, 0), 1))
    colors[i][0] = norm_rgb(r, g, b)


oned_colors = np.zeros(n, dtype=(np.float32, 3))
for i in range(n):
    temperature = t0 + i * dt
    x, y, z = spectrum_to_xyz(bb_spectrum, temperature)
    r, g, b = xyz_to_rgb(SMPTEsystem, x, y, z)
    r = min((max(r, 0), 1))
    g = min((max(g, 0), 1))
    b = min((max(b, 0), 1))
    oned_colors[i] = norm_rgb(r, g, b)


print("color begin: ", colors[0][0], 
      "\ncolor middle: ", colors[n // 2][0],
      "\ncolor end: ",colors[n-1][0])

print("colors: ", colors)

def load_galaxy_star_image():
    raw_image = io.read_png("star-particle.png")
    return raw_image


class Canvas(app.Canvas):

    def __init__(self):
        self.width = 800
        self.height = 600
        app.Canvas.__init__(self, keys='interactive')
        self.size = self.width, self.height

        self._timer = app.Timer('auto', connect=self.update, start=True)

    def __create_galaxy_vertex_data(self):

        data = np.zeros(len(galaxy), 
                        dtype=[('a_size', np.float32, 1),
                  ('a_position', np.float32, 2),
                  #('a_temperature', np.float32, 1),
                  ('a_color_index', np.float32, 1),
                  ('a_brightness', np.float32, 1),
                  ('a_type', np.float32, 1)])


        data['a_size'] = galaxy['size'] * max(self.width/800.0, self.height/800.0)
        data['a_position'] = galaxy['position'] / 13000.0


        data['a_color_index'] = (galaxy['temperature'] - t0) / (t1 - t0)
        #data['a_temperature'] = (galaxy['temperature'] - t0) / (t1 - t0)
        data['a_brightness'] = galaxy['brightness']
        data['a_type'] = galaxy['type']

        return data

    def on_initialize(self, event):

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER, count=len(galaxy))

        self.view = np.eye(4, dtype=np.float32)
        transforms.translate(self.view, 0, 0, -5)

        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        
        #HACK
        self.program['u_colormap'] = oned_colors  # colors

        self.texture = gloo.Texture2D(load_galaxy_star_image(), interpolation='linear')
        self.program['u_texture'] = self.texture

        self.projection = perspective(45.0, self.width/float(self.height), 1.0, 1000.0)
        self.program['u_projection'] = self.projection


        galaxy.update(100000) # in years !
        data = self.__create_galaxy_vertex_data()

        #setup the VBO once the galaxy vertex data has been updated
        self.data_vbo = gloo.VertexBuffer(data)
        self.program.bind(self.data_vbo)


        self.__setup_blending_mode()

    def __setup_blending_mode(self):
        gloo.gl.glClearColor(0.0, 0.0, 0.03, 1.0)
        gloo.gl.glDisable(gloo.gl.GL_DEPTH_TEST)
        gloo.gl.glEnable(gloo.gl.GL_BLEND)
        gloo.gl.glBlendFunc(gloo.gl.GL_SRC_ALPHA, gloo.gl.GL_ONE)

        gloo.gl.glEnable(34370)  # gl.GL_VERTEX_PROGRAM_POINT_SIZE
        gloo.gl.glEnable(34913)  # gl.GL_POINT_SPRITE
        

    def on_resize(self, event):
        self.width, self.height = event.size
        gloo.set_viewport(0, 0, self.width, self.height)
        self.projection = perspective(45.0, self.width / float(self.height), 1.0, 1000.0)
        self.program['u_projection'] = self.projection

       

    def on_draw(self, event):
        #update the galaxy:
        galaxy.update(100000) # in years !


        data = self.__create_galaxy_vertex_data()

        #self.program.bind(gloo.VertexBuffer(data))

        gloo.clear(color=True, depth=True)
        

        #self.data_vbo.set_data(data)
        self.data_vbo = gloo.VertexBuffer(data)
        self.program.bind(self.data_vbo)


        self.program.draw('points')

if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()
