#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import zipfile


class ResultFormat(object):
    def __init__(self, data_type, result_path):
        self.data_type = data_type
        self.result_path = result_path

        if osp.isfile(result_path):
            os.remove(result_path)

        if result_path.endswith('.zip'):
            result_path = result_path.replace('.zip', '')

        if not osp.exists(result_path):
            os.makedirs(result_path)

    def write_result(self, img_name, outputs):
        if 'IC15' in self.data_type or "IC17" in self.data_type:
            self._write_result_ic15(img_name, outputs)
        elif 'TT' in self.data_type:
            self._write_result_tt(img_name, outputs)
        elif 'CTW' in self.data_type:
            self._write_result_ctw(img_name, outputs)
        elif 'MSRA' in self.data_type:
            self._write_result_msra(img_name, outputs)

    def _write_result_ic15(self, img_name, outputs):
        assert self.result_path.endswith('.zip'), 'Error: ic15 result should be a zip file!'

        tmp_folder = self.result_path.replace('.zip', '')

        bboxes = outputs['bboxes']

        lines = []
        for i, bbox in enumerate(bboxes):
            values = [int(v) for v in bbox]
            line = "%d,%d,%d,%d,%d,%d,%d,%d\n" % tuple(values)
            lines.append(line)

        file_name = 'res_%s.txt' % img_name
        file_path = osp.join(tmp_folder, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

        z = zipfile.ZipFile(self.result_path, 'a', zipfile.ZIP_DEFLATED)
        z.write(file_path, file_name)
        z.close()

    def _write_result_tt(self, image_name, outputs):
        bboxes = outputs['bboxes']

        lines = []
        for i, bbox in enumerate(bboxes):
            bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ",%d" % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % image_name
        file_path = osp.join(self.result_path, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

    def _write_result_ctw(self, image_name, outputs):
        bboxes = outputs['bboxes']

        lines = []
        for i, bbox in enumerate(bboxes):
            bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ",%d" % values[v_id]
            line += '\n'
            lines.append(line)

        tmp_folder = self.result_path.replace('.zip', '')

        file_name = '%s.txt' % image_name
        file_path = osp.join(tmp_folder, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)

        z = zipfile.ZipFile(self.result_path, 'a', zipfile.ZIP_DEFLATED)
        z.write(file_path, file_name)
        z.close()




    def _write_result_msra(self, image_name, outputs):
        bboxes = outputs['bboxes']

        lines = []
        for b_idx, bbox in enumerate(bboxes):
            values = [int(v) for v in bbox]
            line = "%d" % values[0]
            for v_id in range(1, len(values)):
                line += ", %d" % values[v_id]
            line += '\n'
            lines.append(line)

        file_name = '%s.txt' % image_name
        file_path = osp.join(self.result_path, file_name)
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line)
