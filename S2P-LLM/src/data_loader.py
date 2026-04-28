import json
import re
import networkx as nx
import os


class CodexDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.id2entity = {}
        self.id2relation = {}
        self.train_triples = []  # (h, r, t) 用于建图
        self.test_samples = []  # (h, r, t, label) 用于测试
        self.graph = nx.MultiDiGraph()

        self.pattern = re.compile(r'\(\s*(.+?),\s*(.+?),\s*(.+?)\s*\)')

    def _parse_file(self, filename, is_test=False):
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found.")
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        for item in data:
            h_id, r_id, t_id = item['embedding_ids']

            # 提取标签 (仅针对测试集)
            label = 1  # 默认为正样本 (Train set usually only has positives)
            if is_test and 'output' in item:
                label = 1 if item['output'] == "True" else 0

            # 提取文本用于 Prompt
            input_str = item['input'].replace('\n', ' ').strip()
            match = self.pattern.search(input_str)

            if match:
                h_text = match.group(1).strip()
                r_text = match.group(2).strip()
                t_text = match.group(3).strip()

                if h_id not in self.id2entity: self.id2entity[h_id] = h_text
                if t_id not in self.id2entity: self.id2entity[t_id] = t_text
                if r_id not in self.id2relation: self.id2relation[r_id] = r_text
            else:
                # Fallback
                if h_id not in self.id2entity: self.id2entity[h_id] = str(h_id)
                if t_id not in self.id2entity: self.id2entity[t_id] = str(t_id)
                if r_id not in self.id2relation: self.id2relation[r_id] = str(r_id)

            if is_test:
                samples.append((h_id, r_id, t_id, label))
            else:
                samples.append((h_id, r_id, t_id))

        return samples

    def load_data(self):
        print("Loading CoDeX-S Datasets...")

        # 1. 加载 Train (仅包含正样本，用于构建图)
        self.train_triples = self._parse_file("CoDeX-S-train.json", is_test=False)

        # 2. 加载 Test (包含 True/False 标签)
        self.test_samples = self._parse_file("CoDeX-S-test.json", is_test=True)

        print(f"Loaded: Train={len(self.train_triples)} triples, Test={len(self.test_samples)} samples")
        print(f"Total Entities: {len(self.id2entity)}, Total Relations: {len(self.id2relation)}")

        # 3. 构建图 (仅使用 Train 数据)
        print("Building Graph from Training Data...")
        for h, r, t in self.train_triples:
            self.graph.add_edge(h, t, relation=r)