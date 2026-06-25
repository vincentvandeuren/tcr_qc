
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Union
import re
from collections import defaultdict
from natsort import natsorted

class BaseMapper():
    def transform(self, key:Union[List[Path], Path]):
        if isinstance(key, list):
            return [self._transform(k) for k in key]
        else:
            return self._transform(key)

    @abstractmethod
    def _transform(self, key):
        pass

    def test(self, dir:List[Path]):
        failed = []
        succes = {}

        for file in dir:
            if file.is_dir():
                continue

            file = file.name
            try:
                succes[file] = self._transform(file)
            except KeyError:
                failed.append(file)

        print(f"Success rate: {len(succes) / (len(succes) + len(failed))}")

        if failed:
            print(f"Failed on {len(failed)} files:")
            print(failed)

        succes_inv = defaultdict(list)
        for k, v in succes.items():
            succes_inv[v].append(k)

        print("Succesfully extracted:")
        for k in natsorted(succes_inv.keys()):
            print(f"{k} <- {', '.join(succes_inv[k])}")

class FileNameMapper(BaseMapper):
    def _transform(self, key:Union[Path, str]):
        if isinstance(key, Path):
            key = key.name
        
        return key
    
    def __repr__(self):
        return "FileNameMapper()"
    
class DictMapper(BaseMapper):
    def __init__(self, mapping:Dict):
        self.mapping = mapping

    def _transform(self, key:Union[Path, str]):
        if isinstance(key, Path):
            key = key.name

        if key not in self.mapping:
            raise KeyError(f"Key {key} not in mapping")
        
        return self.mapping[key]
    
    def __repr__(self):
        return f"DictMapper(mapping={self.mapping})"

class ReplacingDictMapper(BaseMapper):
    def __init__(self, mapping:Dict):
        self.mapping = mapping

    def _transform(self, key:Union[Path, str]):
        if isinstance(key, Path):
            key = key.name

        return self.mapping.get(key, key)
    
    def __repr__(self):
        return f"ReplacingDictMapper(mapping={self.mapping})"
    

class RegexMapper(BaseMapper):
    def __init__(self, pattern:Union[str, re.Pattern], group:Union[int, List[int]]=1):
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self.pattern = pattern
        self.group = group

    def _extract_group(self, m):
        if isinstance(self.group, int):
            return m.group(self.group)
        else:
            return "+".join([m.group(g) if m.group(g) is not None else "" for g in self.group])
        
    def test(self, dir:Path):
        self._explain_groups()
        super().test(dir)

        
    def _transform(self, key):
        if isinstance(key, Path):
            key = key.name
        m = re.search(self.pattern, key)
        
        if m:
            return self._extract_group(m)
        
        raise KeyError(f"Could not match pattern {self.pattern.pattern} in {key}")
    

    def _explain_groups(self):
        if isinstance(self.group, int):
            selected_group = [self.group]
        else:
            selected_group = self.group
    
        groups = extract_groups(self.pattern.pattern)

        print(f"Pattern: {self.pattern.pattern}")

        for g, s in groups:
            if g in selected_group:
                print(f"* group {g}: {s}")
            
            else:
                print(f"  group {g}: {s}")

    def __repr__(self):
        return f"RegexMapper(pattern={self.pattern.pattern}, group={self.group})"
                

class ChainedMapper(BaseMapper):
    def __init__(self, mappers:List[BaseMapper]):
        self.mappers = mappers

    def _transform(self, key:Union[Path, str]):
        if isinstance(key, Path):
            key = key.name

        for mapper in self.mappers:
            key = mapper.transform(key)
        return key
    
    def __repr__(self):
        return f"ChainedMapper(mappers={self.mappers})"

def is_escaped(regex_str, i):
    if i == 0:
        return False
    backslash_count = 0
    j = i - 1
    while j >= 0 and regex_str[j] == '\\':
        backslash_count += 1
        j -= 1
    return backslash_count % 2 == 1

def is_capturing_group(regex_str, i):
    if i + 1 >= len(regex_str):
        return True
    if regex_str[i+1] != '?':
        return True
    j = i + 2
    if j >= len(regex_str):
        return False
    next_char = regex_str[j]
    if next_char == ':':
        return False
    elif next_char == 'P':
        if j + 1 < len(regex_str) and regex_str[j+1] == '<':
            return True
        else:
            return False
    elif next_char in ('=', '!', '<', '>', '#'):
        return False
    elif next_char == '(':
        return False
    else:
        k = j
        valid_flags = {'a', 'i', 'L', 'm', 's', 'u', 'x'}
        while k < len(regex_str) and regex_str[k] != ')':
            c = regex_str[k]
            if c == '-' and k + 1 < len(regex_str) and regex_str[k+1] in valid_flags:
                k += 1
            elif c not in valid_flags:
                break
            k += 1
        if k < len(regex_str) and regex_str[k] == ')':
            return False
        else:
            return False

def extract_groups(regex_str):
    stack = []
    groups = []
    group_number = 0
    i = 0
    n = len(regex_str)
    while i < n:
        if regex_str[i] == '(' and not is_escaped(regex_str, i):
            if is_capturing_group(regex_str, i):
                group_number += 1
                stack.append((group_number, i))
            i += 1
        elif regex_str[i] == ')' and not is_escaped(regex_str, i):
            if stack:
                g_num, start = stack.pop()
                groups.append((g_num, start, i))
            i += 1
        else:
            i += 1
    groups.sort(key=lambda x: x[0])
    result = []
    for g in groups:
        start = g[1]
        end = g[2]
        substring = regex_str[start+1:end]
        result.append((g[0], substring))
    return result
