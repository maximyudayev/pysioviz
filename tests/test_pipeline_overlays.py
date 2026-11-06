"""Unit tests for PysioViz pipeline computer vision overlays."""

import os
from pathlib import Path
import unittest
import numpy as np

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PyQt6 import QtGui, QtWidgets

from pysioviz.qt.pipeline_overlays import (
    DEFAULT_CAMERA_IDS,
    FrameOverlayData,
    PipelineOverlayManager,
)
from pysioviz.qt.widgets.video_grid_dock import VideoCanvasWidget


class TestPipelineOverlays(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication(['unittest'])

    def test_pipeline_overlay_manager_discovery(self):
        workspace_src = Path('c:/Users/maxim/Documents/Code/Frameworks/hermes-gait-analysis/src')
        if not workspace_src.exists():
            self.skipTest('hermes-gait-analysis workspace not found on disk.')

        mgr = PipelineOverlayManager.discover(workspace_src)
        self.assertIsNotNone(mgr)
        self.assertGreater(len(mgr._frame_data), 0)

        # Test querying cam 40478064 frame 2260 (where subject is present)
        cam_id = str(DEFAULT_CAMERA_IDS[0])
        data = mgr.get_overlay_data(cam_id, 2260)
        self.assertIsNotNone(data)
        self.assertIsInstance(data, FrameOverlayData)
        self.assertEqual(data.cam_id, cam_id)
        self.assertTrue(data.subj_present)
        self.assertIsNotNone(data.bbox_xyxy)
        self.assertEqual(len(data.bbox_xyxy), 4)
        self.assertIsNotNone(data.keypoints_2d)

        # Test mask and bounds retrieval (fast Format_Indexed8 path)
        mask_info = mgr.get_mask_and_bounds(cam_id, data.v_idx)
        self.assertIsNotNone(mask_info)
        qimg, xmin, ymin, side = mask_info
        self.assertIsInstance(qimg, QtGui.QImage)
        self.assertEqual(qimg.width(), 1024)
        self.assertEqual(qimg.height(), 1024)
        self.assertGreater(side, 0)

        # Clean shutdown of background prefetch worker
        mgr.stop()

    def test_video_canvas_overlay_rendering(self):
        canvas = VideoCanvasWidget()
        canvas.resize(640, 360)

        # Mock video frame
        frame_img = QtGui.QImage(1920, 1080, QtGui.QImage.Format.Format_RGB888)
        frame_img.fill(QtGui.QColor(25, 25, 30))
        canvas.set_pixmap(QtGui.QPixmap.fromImage(frame_img))

        # Mock overlay
        overlay = FrameOverlayData(
            cam_id='40478064',
            frame_idx=100,
            t_mid_sec=10.0,
            subj_present=True,
            reid_cosine_sim=0.98,
            bbox_xyxy=(500.0, 200.0, 900.0, 800.0),
            v_idx=0,
            keypoints_2d=np.array([[700.0, 300.0], [700.0, 500.0]], dtype=np.float32),
            keypoint_confs=np.array([0.95, 0.90], dtype=np.float32),
        )

        # Mock mask
        mask_img = QtGui.QImage(1024, 1024, QtGui.QImage.Format.Format_RGBA8888)
        mask_img.fill(QtGui.QColor(0, 245, 160, 90))
        mask_info = (mask_img, 450.0, 150.0, 700.0)

        canvas.set_overlay(overlay, mask_info)

        # Render to target pixmap
        target = QtGui.QPixmap(640, 360)
        canvas.render(target)
        self.assertFalse(target.isNull())

        # Toggle options
        canvas.set_overlay_visibility(show_bbox=False, show_mask=False, show_pose=False, show_reid_label=False)
        canvas.render(target)
        self.assertFalse(target.isNull())


if __name__ == '__main__':
    unittest.main()
