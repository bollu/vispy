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

//sampler that maps [0, n] -> color according to blackbody law
uniform sampler1D u_colormap;
//index to sample the colormap at
attribute float a_color_index;

//size of the star
attribute float a_size;
//type
//type 0 - stars
//type 1 - dust
//type 2 - h2a objects
//type 3 - h2a objects
attribute float a_type;
attribute vec2  a_position;
//brightness of the star
attribute float a_brightness;

varying vec3 v_color;
void main (void)
{   
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 0.0,1.0);

    //find base color according to physics from our sampler
    vec3 base_color = texture1D(u_colormap, a_color_index).rgb;
    //scale it down according to brightness
    v_color = base_color * a_brightness;


    if (a_size > 2.0)
    {
        gl_PointSize = a_size;
    } else {
        gl_PointSize = 0.0;
    }

    if (a_type == 2) {
        v_color *= vec3(2, 1, 1);
    }
    else if (a_type == 3) {
        v_color = vec3(.9);
    }
}
"""

FRAG_SHADER = """
#version 120
//star texture
uniform sampler2D u_texture;
//predicted color from black body
varying vec3 v_color;

void main()
{
    //amount of intensity from the grayscale star
    float star_tex_intensity = texture2D(u_texture, gl_PointCoord).r;
    gl_FragColor = vec4(v_color * star_tex_intensity, 0.8);
}
"""

galaxy = Galaxy(35000)
galaxy.reset(13000, 4000, 0.0004, 0.90, 0.90, 0.5, 200, 300)
# coldest and hottest temperatures of out galaxy
t0, t1 = 800.0, 10000.0
# total number of discrete colors between t0 and t1
n = 156
dt = (t1 - t0) / n

# maps [0, n) -> colors
# generate a linear interpolation of temperatures
# then map the temperatures to colors using black body
# color predictions
colors = np.zeros(n, dtype=(np.float32, 3))
for i in range(n):
    temperature = t0 + i * dt
    x, y, z = spectrum_to_xyz(bb_spectrum, temperature)
    r, g, b = xyz_to_rgb(SMPTEsystem, x, y, z)
    r = min((max(r, 0), 1))
    g = min((max(g, 0), 1))
    b = min((max(b, 0), 1))
    colors[i] = norm_rgb(r, g, b)


# load the PNG that we use to blend the star with
# to provide a circular look to each star.
def load_galaxy_star_image():
    raw_image = io.read_png("star-particle.png")
    return raw_image


class Canvas(app.Canvas):

    def __init__(self):
        # setup initial width, height
        self.width = 800
        self.height = 600
        app.Canvas.__init__(self, keys='interactive')
        self.size = self.width, self.height

        self._timer = app.Timer('auto', connect=self.update, start=True)

    # create the numpy array corresponding to the shader data
    def __create_galaxy_vertex_data(self):
        data = np.zeros(len(galaxy),
                        dtype=[('a_size', np.float32, 1),
                               ('a_position', np.float32, 2),
                               ('a_color_index', np.float32, 1),
                               ('a_brightness', np.float32, 1),
                               ('a_type', np.float32, 1)])

        # see shader for parameter explanations
        data['a_size'] = galaxy['size'] * max(self.width / 800.0,
                                              self.height / 800.0)
        data['a_position'] = galaxy['position'] / 13000.0

        data['a_color_index'] = (galaxy['temperature'] - t0) / (t1 - t0)
        data['a_brightness'] = galaxy['brightness']
        data['a_type'] = galaxy['type']

        return data

    def on_initialize(self, event):
        # create a new shader program
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER,
                                    count=len(galaxy))

        #load the star texture
        self.texture = gloo.Texture2D(load_galaxy_star_image(),
                                      interpolation='linear')
        self.program['u_texture'] = self.texture

        # construct the model, view and projection matrices
        self.view = np.eye(4, dtype=np.float32)
        transforms.translate(self.view, 0, 0, -5)
        self.program['u_view'] = self.view

        self.model = np.eye(4, dtype=np.float32)
        self.program['u_model'] = self.model

        self.program['u_colormap'] = colors

        self.projection = perspective(45.0, self.width / float(self.height),
                                      1.0, 1000.0)
        self.program['u_projection'] = self.projection

        # start the galaxy to some decent point in the future
        galaxy.update(100000)
        data = self.__create_galaxy_vertex_data()

        # setup the VBO once the galaxy vertex data has been setup
        # bind the VBO for the first time
        self.data_vbo = gloo.VertexBuffer(data)
        self.program.bind(self.data_vbo)

        # setup blending
        self.__setup_blending_mode()

    def __setup_blending_mode(self):
        # set the clear color to almost black
        gloo.gl.glClearColor(0.0, 0.0, 0.03, 1.0)
        gloo.gl.glDisable(gloo.gl.GL_DEPTH_TEST)
        # use additive blend mode so that the
        # particles overlap and intensify
        gloo.gl.glEnable(gloo.gl.GL_BLEND)
        gloo.gl.glBlendFunc(gloo.gl.GL_SRC_ALPHA, gloo.gl.GL_ONE)

    def on_resize(self, event):
        self.width, self.height = event.size
        # setup the new viewport
        gloo.set_viewport(0, 0, self.width, self.height)
        # recompute the projection matrix
        self.projection = perspective(45.0, self.width / float(self.height),
                                      1.0, 1000.0)
        self.program['u_projection'] = self.projection

    def on_draw(self, event):
        # update the galaxy
        galaxy.update(100000)  # in years !

        # recreate the numpy array that will be sent as the VBO data
        data = self.__create_galaxy_vertex_data()
        # update the VBO
        self.data_vbo.set_data(data)
        # bind the VBO to the GL context
        self.program.bind(self.data_vbo)

        # clear the screen and render
        gloo.clear(color=True, depth=True)
        self.program.draw('points')

if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()
