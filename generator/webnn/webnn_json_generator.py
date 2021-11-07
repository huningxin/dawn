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
#from dawn_json_generator import parse_json, Method, Name


############################################################
# OBJECT MODEL
############################################################


class Name:
    def __init__(self, name, native=False):
        self.native = native
        self.name = name
        if native:
            self.chunks = [name]
        else:
            self.chunks = name.split(' ')

    def get(self):
        return self.name

    def CamelChunk(self, chunk):
        return chunk[0].upper() + chunk[1:]

    def canonical_case(self):
        return (' '.join(self.chunks)).lower()

    def concatcase(self):
        return ''.join(self.chunks)

    def camelCase(self):
        return self.chunks[0] + ''.join(
            [self.CamelChunk(chunk) for chunk in self.chunks[1:]])

    def CamelCase(self):
        return ''.join([self.CamelChunk(chunk) for chunk in self.chunks])

    def SNAKE_CASE(self):
        return '_'.join([chunk.upper() for chunk in self.chunks])

    def snake_case(self):
        return '_'.join(self.chunks)

    def js_enum_case(self):
        result = self.chunks[0].lower()
        for chunk in self.chunks[1:]:
            if not result[-1].isdigit():
                result += '-'
            result += chunk.lower()
        return result


def concat_names(*names):
    return ' '.join([name.canonical_case() for name in names])


class Type:
    def __init__(self, name, json_data, native=False):
        self.json_data = json_data
        self.dict_name = name
        self.name = Name(name, native=native)
        self.category = json_data['category']
        self.javascript = self.json_data.get('javascript', True)


EnumValue = namedtuple('EnumValue', ['name', 'value', 'valid', 'jsrepr'])


class EnumType(Type):
    def __init__(self, name, json_data):
        Type.__init__(self, name, json_data)

        self.values = []
        self.contiguousFromZero = True
        lastValue = -1
        for m in self.json_data['values']:
            value = m['value']
            if value != lastValue + 1:
                self.contiguousFromZero = False
            lastValue = value
            self.values.append(
                EnumValue(Name(m['name']), value, m.get('valid', True),
                          m.get('jsrepr', None)))

        # Assert that all values are unique in enums
        all_values = set()
        for value in self.values:
            if value.value in all_values:
                raise Exception("Duplicate value {} in enum {}".format(
                    value.value, name))
            all_values.add(value.value)


BitmaskValue = namedtuple('BitmaskValue', ['name', 'value'])


class BitmaskType(Type):
    def __init__(self, name, json_data):
        Type.__init__(self, name, json_data)
        self.values = [
            BitmaskValue(Name(m['name']), m['value'])
            for m in self.json_data['values']
        ]
        self.full_mask = 0
        for value in self.values:
            self.full_mask = self.full_mask | value.value


class CallbackType(Type):
    def __init__(self, name, json_data):
        Type.__init__(self, name, json_data)
        self.arguments = []


class NativeType(Type):
    def __init__(self, name, json_data):
        Type.__init__(self, name, json_data, native=True)


# Methods and structures are both "records", so record members correspond to
# method arguments or structure members.
class RecordMember:
    def __init__(self,
                 name,
                 typ,
                 annotation,
                 optional=False,
                 is_return_value=False,
                 default_value=None,
                 skip_serialize=False):
        self.name = name
        self.type = typ
        self.annotation = annotation
        self.length = None
        self.optional = optional
        self.is_return_value = is_return_value
        self.handle_type = None
        self.default_value = default_value
        self.skip_serialize = skip_serialize

    def set_handle_type(self, handle_type):
        assert self.type.dict_name == "ObjectHandle"
        self.handle_type = handle_type


Method = namedtuple('Method', ['name', 'return_type', 'arguments'])


class ObjectType(Type):
    def __init__(self, name, json_data):
        Type.__init__(self, name, json_data)
        self.methods = []
        self.built_type = None


class Record:
    def __init__(self, name):
        self.name = Name(name)
        self.members = []
        self.may_have_dawn_object = False

    def update_metadata(self):
        def may_have_dawn_object(member):
            if isinstance(member.type, ObjectType):
                return True
            elif isinstance(member.type, StructureType):
                return member.type.may_have_dawn_object
            else:
                return False

        self.may_have_dawn_object = any(
            may_have_dawn_object(member) for member in self.members)

        # Set may_have_dawn_object to true if the type is chained or
        # extensible. Chained structs may contain a Dawn object.
        if isinstance(self, StructureType):
            self.may_have_dawn_object = (self.may_have_dawn_object
                                         or self.chained or self.extensible)


class StructureType(Record, Type):
    def __init__(self, name, json_data):
        Record.__init__(self, name)
        Type.__init__(self, name, json_data)
        self.chained = json_data.get("chained", False)
        self.extensible = json_data.get("extensible", False)
        # Chained structs inherit from wgpu::ChainedStruct, which has
        # nextInChain, so setting both extensible and chained would result in
        # two nextInChain members.
        assert not (self.extensible and self.chained)


class Command(Record):
    def __init__(self, name, members=None):
        Record.__init__(self, name)
        self.members = members or []
        self.derived_object = None
        self.derived_method = None


def linked_record_members(json_data, types):
    members = []
    members_by_name = {}
    for m in json_data:
        member = RecordMember(Name(m['name']),
                              types[m['type']],
                              m.get('annotation', 'value'),
                              optional=m.get('optional', False),
                              is_return_value=m.get('is_return_value', False),
                              default_value=m.get('default', None),
                              skip_serialize=m.get('skip_serialize', False))
        handle_type = m.get('handle_type')
        if handle_type:
            member.set_handle_type(types[handle_type])
        members.append(member)
        members_by_name[member.name.canonical_case()] = member

    for (member, m) in zip(members, json_data):
        if member.annotation != 'value':
            if not 'length' in m:
                if member.type.category != 'object':
                    member.length = "constant"
                    member.constant_length = 1
                else:
                    assert False
            elif m['length'] == 'strlen':
                member.length = 'strlen'
            else:
                member.length = members_by_name[m['length']]

    return members


############################################################
# PARSE
############################################################


def link_object(obj, types):
    def make_method(json_data):
        arguments = linked_record_members(json_data.get('args', []), types)
        return Method(Name(json_data['name']),
                      types[json_data.get('returns', 'void')], arguments)

    obj.methods = [make_method(m) for m in obj.json_data.get('methods', [])]
    obj.methods.sort(key=lambda method: method.name.canonical_case())


def link_structure(struct, types):
    struct.members = linked_record_members(struct.json_data['members'], types)


def link_callback(callback, types):
    callback.arguments = linked_record_members(callback.json_data['args'],
                                               types)


# Sort structures so that if struct A has struct B as a member, then B is
# listed before A.
#
# This is a form of topological sort where we try to keep the order reasonably
# similar to the original order (though the sort isn't technically stable).
#
# It works by computing for each struct type what is the depth of its DAG of
# dependents, then resorting based on that depth using Python's stable sort.
# This makes a toposort because if A depends on B then its depth will be bigger
# than B's. It is also nice because all nodes with the same depth are kept in
# the input order.
def topo_sort_structure(structs):
    for struct in structs:
        struct.visited = False
        struct.subdag_depth = 0

    def compute_depth(struct):
        if struct.visited:
            return struct.subdag_depth

        max_dependent_depth = 0
        for member in struct.members:
            if member.type.category == 'structure':
                max_dependent_depth = max(max_dependent_depth,
                                          compute_depth(member.type) + 1)

        struct.subdag_depth = max_dependent_depth
        struct.visited = True
        return struct.subdag_depth

    for struct in structs:
        compute_depth(struct)

    result = sorted(structs, key=lambda struct: struct.subdag_depth)

    for struct in structs:
        del struct.visited
        del struct.subdag_depth

    return result


def parse_json(json):
    category_to_parser = {
        'bitmask': BitmaskType,
        'enum': EnumType,
        'native': NativeType,
        'callback': CallbackType,
        'object': ObjectType,
        'structure': StructureType,
    }

    types = {}

    by_category = {}
    for name in category_to_parser.keys():
        by_category[name] = []

    for (name, json_data) in json.items():
        if name[0] == '_':
            continue
        category = json_data['category']
        parsed = category_to_parser[category](name, json_data)
        by_category[category].append(parsed)
        types[name] = parsed

    for obj in by_category['object']:
        link_object(obj, types)

    for struct in by_category['structure']:
        link_structure(struct, types)

    for callback in by_category['callback']:
        link_callback(callback, types)

    for category in by_category.keys():
        by_category[category] = sorted(
            by_category[category], key=lambda typ: typ.name.canonical_case())

    by_category['structure'] = topo_sort_structure(by_category['structure'])

    for struct in by_category['structure']:
        struct.update_metadata()

    return {'types': types, 'by_category': by_category}


#############################################################
# Generator
#############################################################


def as_varName(*names):
    varName = names[0].camelCase() + ''.join(
        [name.CamelCase() for name in names[1:]])
    # Avoid to use C++ keyword 'operator', probably check for others on demand.
    if varName == 'operator':
        varName = 'mlOperator'
    return varName


def as_cType(name):
    if name.native:
        return name.concatcase()
    else:
        return 'ML' + name.CamelCase()


def as_cTypeDawn(name):
    if name.native:
        return name.concatcase()
    else:
        return 'Dawn' + name.CamelCase()


def as_cTypeEnumSpecialCase(typ):
    if typ.category == 'bitmask':
        return as_cType(typ.name) + 'Flags'
    return as_cType(typ.name)


def as_cppType(name):
    if name.native:
        return name.concatcase()
    else:
        return name.CamelCase()


def as_jsEnumValue(value):
    if value.jsrepr: return value.jsrepr
    return "'" + value.name.js_enum_case() + "'"


def convert_cType_to_cppType(typ, annotation, arg, indent=0):
    if typ.category == 'native':
        return arg
    if annotation == 'value':
        if typ.category == 'object':
            return '{}::Acquire({})'.format(as_cppType(typ.name), arg)
        elif typ.category == 'structure':
            converted_members = [
                convert_cType_to_cppType(
                    member.type, member.annotation,
                    '{}.{}'.format(arg, as_varName(member.name)), indent + 1)
                for member in typ.members
            ]

            converted_members = [(' ' * 4) + m for m in converted_members]
            converted_members = ',\n'.join(converted_members)

            return as_cppType(typ.name) + ' {\n' + converted_members + '\n}'
        else:
            return 'static_cast<{}>({})'.format(as_cppType(typ.name), arg)
    else:
        return 'reinterpret_cast<{} {}>({})'.format(as_cppType(typ.name),
                                                    annotation, arg)


def decorate(name, typ, arg):
    if arg.annotation == 'value':
        return typ + ' ' + name
    elif arg.annotation == '*':
        return typ + ' * ' + name
    elif arg.annotation == 'const*':
        return typ + ' const * ' + name
    elif arg.annotation == 'const*const*':
        return 'const ' + typ + '* const * ' + name
    else:
        assert False


def annotated(typ, arg):
    name = as_varName(arg.name)
    return decorate(name, typ, arg)


def as_cEnum(type_name, value_name):
    assert not type_name.native and not value_name.native
    return 'ML' + type_name.CamelCase() + '_' + value_name.CamelCase()


def as_cEnumDawn(type_name, value_name):
    assert not type_name.native and not value_name.native
    return ('DAWN' + '_' + type_name.SNAKE_CASE() + '_' +
            value_name.SNAKE_CASE())


def as_cppEnum(value_name):
    assert not value_name.native
    if value_name.concatcase()[0].isdigit():
        return "e" + value_name.CamelCase()
    return value_name.CamelCase()


def as_cMethod(type_name, method_name):
    assert not type_name.native and not method_name.native
    return 'ml' + type_name.CamelCase() + method_name.CamelCase()


def as_cMethodDawn(type_name, method_name):
    assert not type_name.native and not method_name.native
    return 'dawn' + type_name.CamelCase() + method_name.CamelCase()


def as_MethodSuffix(type_name, method_name):
    assert not type_name.native and not method_name.native
    return type_name.CamelCase() + method_name.CamelCase()


def as_cProc(type_name, method_name):
    assert not type_name.native and not method_name.native
    return 'Webnn' + 'Proc' + type_name.CamelCase() + method_name.CamelCase()


def as_cProcDawn(type_name, method_name):
    assert not type_name.native and not method_name.native
    return 'Dawn' + 'Proc' + type_name.CamelCase() + method_name.CamelCase()


def as_frontendType(typ):
    if typ.category == 'object':
        return typ.name.CamelCase() + 'Base*'
    elif typ.category in ['bitmask', 'enum']:
        return 'ml::' + typ.name.CamelCase()
    elif typ.category == 'structure':
        return as_cppType(typ.name)
    else:
        return as_cType(typ.name)


def c_methods(types, typ):
    return typ.methods + [
        Method(Name('reference'), types['void'], []),
        Method(Name('release'), types['void'], []),
    ]


def get_c_methods_sorted_by_name(api_params):
    unsorted = [(as_MethodSuffix(typ.name, method.name), typ, method) \
            for typ in api_params['by_category']['object'] \
            for method in c_methods(api_params['types'], typ) ]
    return [(typ, method) for (_, typ, method) in sorted(unsorted)]


def has_callback_arguments(method):
    return any(arg.type.category == 'callback' for arg in method.arguments)


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
        api_params = parse_json(loaded_json)

        targets = args.targets.split(',')

        base_params = {
            'Name': lambda name: Name(name),
            'as_annotated_cType': \
                lambda arg: annotated(as_cTypeEnumSpecialCase(arg.type), arg),
            'as_annotated_cppType': \
                lambda arg: annotated(as_cppType(arg.type.name), arg),
            'as_cEnum': as_cEnum,
            'as_cEnumDawn': as_cEnumDawn,
            'as_cppEnum': as_cppEnum,
            'as_cMethod': as_cMethod,
            'as_cMethodDawn': as_cMethodDawn,
            'as_MethodSuffix': as_MethodSuffix,
            'as_cProc': as_cProc,
            'as_cProcDawn': as_cProcDawn,
            'as_cType': as_cType,
            'as_cTypeDawn': as_cTypeDawn,
            'as_cppType': as_cppType,
            'as_jsEnumValue': as_jsEnumValue,
            'convert_cType_to_cppType': convert_cType_to_cppType,
            'as_varName': as_varName,
            'decorate': decorate,
            'c_methods': lambda typ: c_methods(api_params['types'], typ),
            'c_methods_sorted_by_name': \
                get_c_methods_sorted_by_name(api_params),
        }

        renders = []

        if 'webnn_headers' in targets:
            renders.append(
                FileRender('webnn.h', 'src/include/dawn/webnn/webnn.h',
                           [base_params, api_params]))
            renders.append(
                FileRender('webnn_proc_table.h',
                           'src/include/dawn/webnn/webnn_proc_table.h',
                           [base_params, api_params]))

        if 'webnncpp_headers' in targets:
            renders.append(
                FileRender('webnn_cpp.h', 'src/include/dawn/webnn/webnn_cpp.h',
                           [base_params, api_params]))

        if 'webnn_proc' in targets:
            renders.append(
                FileRender('webnn_proc.c', 'src/dawn/webnn/webnn_proc.c',
                           [base_params, api_params]))

        if 'webnncpp' in targets:
            renders.append(
                FileRender('webnn_cpp.cpp', 'src/dawn/webnn/webnn_cpp.cpp',
                           [base_params, api_params]))

        if 'emscripten_bits' in targets:
            renders.append(
                FileRender('webnn_struct_info.json',
                           'src/dawn/webnn/webnn_struct_info.json',
                           [base_params, api_params]))
            renders.append(
                FileRender('library_webnn_enum_tables.js',
                           'src/dawn/webnn/library_webnn_enum_tables.js',
                           [base_params, api_params]))

        if 'mock_webnn' in targets:
            mock_params = [
                base_params, api_params, {
                    'has_callback_arguments': has_callback_arguments
                }
            ]
            renders.append(
                FileRender('mock_webnn.h', 'src/dawn/webnn/mock_webnn.h',
                           mock_params))
            renders.append(
                FileRender('mock_webnn.cpp', 'src/dawn/webnn/mock_webnn.cpp',
                           mock_params))

        if 'webnn_native_utils' in targets:
            frontend_params = [
                base_params,
                api_params,
                {
                    # TODO: as_frontendType and co. take a Type, not a Name :(
                    'as_frontendType': lambda typ: as_frontendType(typ),
                    'as_annotated_frontendType': \
                        lambda arg: annotated(as_frontendType(arg.type), arg),
                }
            ]

            renders.append(
                FileRender('webnn_native/ValidationUtils.h',
                           'src/dawn_native/webnn/ValidationUtils_autogen.h',
                           frontend_params))
            renders.append(
                FileRender('webnn_native/ValidationUtils.cpp',
                           'src/dawn_native/webnn/ValidationUtils_autogen.cpp',
                           frontend_params))
            renders.append(
                FileRender('webnn_native/webnn_structs.h',
                           'src/dawn_native/webnn/webnn_structs_autogen.h',
                           frontend_params))
            renders.append(
                FileRender('webnn_native/webnn_structs.cpp',
                           'src/dawn_native/webnn/webnn_structs_autogen.cpp',
                           frontend_params))
            renders.append(
                FileRender('webnn_native/ProcTable.cpp',
                           'src/dawn_native/webnn/ProcTable.cpp', frontend_params))

        return renders

    def get_dependencies(self, args):
        deps = [os.path.abspath(args.webnn_json)]
        return deps


if __name__ == '__main__':
    sys.exit(run_generator(MultiGeneratorFromWebnnJSON()))
