#!/usr/bin/env python3
# Copyright 2017 The Dawn Authors
# Copyright 2021 The WebNN-native Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json, os, sys
from collections import namedtuple

# sys.path.insert(1, 'third_party/dawn/generator')
kDawnGeneratorPath = '--dawn-generator-path'
try:
    dawn_generator_path_argv_index = sys.argv.index(kDawnGeneratorPath)
    path = sys.argv[dawn_generator_path_argv_index + 1]
    sys.path.insert(1, path)
except ValueError:
    # --dawn-generator-path isn't passed, ignore the exception and just import
    # assuming it already is in the Python PATH.
    print('No dawn generator path defined')
    sys.exit(1)

from generator_lib import Generator, run_generator, FileRender
from dawn_json_generator import make_base_render_params, parse_json, \
                            has_callback_arguments, annotated, as_frontendType,\
                            compute_wire_params, as_wireType

class MultiGeneratorFromWebnnJSON(Generator):
    def get_description(self):
        return 'Generates code for various target from Dawn.json.'

    def add_commandline_arguments(self, parser):
        allowed_targets = [
            'webnn_headers', 'webnncpp_headers', 'webnncpp', 'webnn_proc',
            'mock_webnn', 'webnn_native_utils'
        ]

        parser.add_argument('--webnn-json',
                            required=True,
                            type=str,
                            help='The WebNN JSON definition to use.')
        parser.add_argument(
            '--targets',
            required=True,
            type=str,
            help=
            'Comma-separated subset of targets to output. Available targets: '
            + ', '.join(allowed_targets))
        parser.add_argument(
            '--dawn-generator-path',
            required=True,
            type=str,
            help='The path of Dawn generator')

    def get_file_renders(self, args):
        with open(args.webnn_json) as f:
            loaded_json = json.loads(f.read())
        api_params = parse_json(loaded_json, enabled_tags=['dawn', 'native', 'deprecated'])

        targets = args.targets.split(',')

        metadata = api_params['metadata']
        base_params = make_base_render_params(metadata)

        renders = []

        api_file_name = metadata.api.lower()
        proc_table_prefix = metadata.proc_table_prefix.lower()

        if 'webnn_headers' in targets:
            renders.append(
                FileRender('../../templates/api.h', 'src/include/dawn/webnn.h',
                           [base_params, api_params]))
            renders.append(
                FileRender('../../templates/dawn_proc_table.h',
                           'src/include/dawn/' + proc_table_prefix + '_proc_table.h',
                           [base_params, api_params]))

        if 'webnncpp_headers' in targets:
            renders.append(
                FileRender('../../templates/api_cpp.h', 'src/include/dawn/' + api_file_name + '_cpp.h',
                           [base_params, api_params]))

        if 'webnn_proc' in targets:
            renders.append(
                FileRender('../../templates/dawn_proc.c', 'src/dawn/' + proc_table_prefix + '_proc.c',
                           [base_params, api_params]))

        if 'webnncpp' in targets:
            renders.append(
                FileRender('webnn_cpp.cpp', 'src/dawn/webnn_cpp.cpp',
                           [base_params, api_params]))

        if 'emscripten_bits' in targets:
            renders.append(
                FileRender('webnn_struct_info.json',
                           'src/dawn/webnn_struct_info.json',
                           [base_params, api_params]))
            renders.append(
                FileRender('library_webnn_enum_tables.js',
                           'src/dawn/library_webnn_enum_tables.js',
                           [base_params, api_params]))

        if 'mock_webnn' in targets:
            mock_params = [
                base_params, api_params, {
                    'has_callback_arguments': has_callback_arguments
                }
            ]
            renders.append(
                FileRender('mock_webnn.h', 'src/dawn/mock_webnn.h',
                           mock_params))
            renders.append(
                FileRender('mock_webnn.cpp', 'src/dawn/mock_webnn.cpp',
                           mock_params))

        if 'webnn_native_utils' in targets:
            frontend_params = [
                base_params,
                api_params,
                {
                    # TODO: as_frontendType and co. take a Type, not a Name :(
                    'as_frontendType': lambda typ: as_frontendType(metadata, typ),
                    'as_annotated_frontendType': \
                        lambda arg: annotated(as_frontendType(metadata, arg.type), arg),
                }
            ]

            renders.append(
                FileRender('webnn_native/ValidationUtils.h',
                           'src/dawn_native/WebnnValidationUtils_autogen.h',
                           frontend_params))
            renders.append(
                FileRender('webnn_native/ValidationUtils.cpp',
                           'src/dawn_native/WebnnValidationUtils_autogen.cpp',
                           frontend_params))
            renders.append(
                FileRender('webnn_native/webnn_structs.h',
                           'src/dawn_native/webnn_structs_autogen.h',
                           frontend_params))
            renders.append(
                FileRender('webnn_native/webnn_structs.cpp',
                           'src/dawn_native/webnn_structs_autogen.cpp',
                           frontend_params))
            renders.append(
                FileRender('webnn_native/ProcTable.cpp',
                           'src/dawn_native/WebnnProcTable.cpp', frontend_params))

        return renders

    def get_dependencies(self, args):
        deps = [os.path.abspath(args.webnn_json)]
        return deps


if __name__ == '__main__':
    sys.exit(run_generator(MultiGeneratorFromWebnnJSON()))
