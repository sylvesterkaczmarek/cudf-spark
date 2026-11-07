# Copyright (c) 2026, NVIDIA CORPORATION.
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

"""Legacy marker contracts through the real collection and setup hooks."""

from types import SimpleNamespace
import warnings

import pytest

import conftest


class _Item:
    def __init__(self, *marks):
        self._marks = marks
        self._nodeid = 'synthetic_test.py::test_marker'

    @property
    def nodeid(self):
        return self._nodeid

    def get_closest_marker(self, name):
        return next((mark for mark in self._marks if mark.name == name), None)

    def iter_markers(self, name):
        return (mark for mark in self._marks if mark.name == name)


def _mark(name, *args, **kwargs):
    return getattr(pytest.mark, name)(*args, **kwargs).mark


def _process(item, phase):
    if phase == 'setup':
        conftest.pytest_runtest_setup(item)
    else:
        config = SimpleNamespace(getoption=lambda name: 'none')
        conftest.pytest_collection_modifyitems(config, [item])


@pytest.fixture(autouse=True)
def isolated_marker_state(monkeypatch):
    monkeypatch.setattr(conftest, '_is_precommit_run', False)
    monkeypatch.setattr(conftest, '_random_select_config', None)
    conftest.pytest_runtest_setup(_Item())
    yield
    conftest.pytest_runtest_setup(_Item())


@pytest.mark.parametrize('runtime', ['apache', 'databricks'])
@pytest.mark.parametrize('marker', ['allow_non_gpu', 'allow_non_gpu_databricks'])
@pytest.mark.parametrize('phase', ['collection', 'setup'])
@pytest.mark.parametrize('kwargs', [
    {'any': 1}, {'any': 0}, {'any': 'false'}, {'any': None}, {'any': []},
    {'reason': 'test'}, {'anny': True}, {'any': True, 'reason': 'test'},
])
def test_invalid_kwargs_raise_typeerror(monkeypatch, runtime, marker, phase, kwargs):
    monkeypatch.setattr(conftest, '_runtime_env', runtime)
    item = _Item(_mark(marker, 'ProjectExec', **kwargs))

    with pytest.raises(TypeError, match=marker):
        _process(item, phase)

    assert not conftest.is_allowing_any_non_gpu()
    assert list(conftest.get_non_gpu_allowed()) == []


@pytest.mark.parametrize('runtime', ['apache', 'databricks'])
@pytest.mark.parametrize('marker', ['allow_non_gpu', 'allow_non_gpu_databricks'])
@pytest.mark.parametrize('kwargs', [{}, {'any': False}, {'any': True}])
def test_valid_kwargs_preserve_allowances(monkeypatch, runtime, marker, kwargs):
    monkeypatch.setattr(conftest, '_runtime_env', runtime)
    item = _Item(_mark(marker, 'ProjectExec', **kwargs))
    _process(item, 'collection')
    _process(item, 'setup')

    active = marker == 'allow_non_gpu' or runtime == 'databricks'
    allow_any = active and kwargs.get('any', False)
    assert conftest.is_allowing_any_non_gpu() is allow_any
    assert list(conftest.get_non_gpu_allowed()) == (
        ['ProjectExec'] if active and not allow_any else [])


@pytest.mark.parametrize('runtime', ['apache', 'databricks'])
@pytest.mark.parametrize('marker', ['allow_non_gpu', 'allow_non_gpu_databricks'])
def test_empty_marker_keeps_existing_warning(monkeypatch, runtime, marker):
    monkeypatch.setattr(conftest, '_runtime_env', runtime)
    item = _Item(_mark(marker, any=False))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _process(item, 'setup')
    active = marker == 'allow_non_gpu' or runtime == 'databricks'
    assert len(caught) == int(active)
    if active:
        assert str(caught[0].message) == marker + ' marker without anything allowed'
    assert not conftest.is_allowing_any_non_gpu()


@pytest.mark.parametrize('runtime', ['apache', 'databricks'])
@pytest.mark.parametrize('any_value', [False, True])
def test_base_databricks_and_conditional_markers_still_combine(
        monkeypatch, runtime, any_value):
    monkeypatch.setattr(conftest, '_runtime_env', runtime)
    item = _Item(
        _mark('allow_non_gpu', 'BaseExec', any=False),
        _mark('allow_non_gpu_databricks', 'DbExec', any=any_value),
        _mark('allow_non_gpu_conditional', True, 'ConditionalExec'),
    )
    _process(item, 'collection')
    _process(item, 'setup')
    assert conftest.is_allowing_any_non_gpu() is (
        runtime == 'databricks' and any_value)
    expected = ['BaseExec']
    if runtime == 'databricks' and not any_value:
        expected.append('DbExec')
    expected.append('ConditionalExec')
    assert list(conftest.get_non_gpu_allowed()) == expected

    conftest.pytest_runtest_setup(_Item())
    assert not conftest.is_allowing_any_non_gpu()
    assert list(conftest.get_non_gpu_allowed()) == []
