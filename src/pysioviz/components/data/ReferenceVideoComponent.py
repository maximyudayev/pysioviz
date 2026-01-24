############
#
# Copyright (c) 2026 Maxim Yudayev and KU Leuven eMedia Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Created 2024-2026 for the KU Leuven AidWear, AidFOG, and RevalExo projects
# by Maxim Yudayev [https://yudayev.com].
#
# ############

from dash import Output, Input

from pysioviz.components.data import VideoComponent
from pysioviz.utils.gui_utils import app
from pysioviz.utils.types import GlobalVariableId


class ReferenceVideoComponent(VideoComponent):
    def activate_callbacks(self):
        @app.callback(
            Output('%s-video' % (self._unique_id), 'figure'),
            Output('%s-timestamp' % (self._unique_id), 'children'),
            Input(GlobalVariableId.REF_FRAME_TIMESTAMP.value, 'data'),
        )
        def update_camera(ref_timestamp):
            # Reference cameras override camera callback to match frame by closest `frame_timestamp` among each other, selected by frame slider.
            self._current_frame_id = self.get_frame_for_timestamp(ref_timestamp)
            return self._generate_patch_from_frame(self._current_frame_id)
