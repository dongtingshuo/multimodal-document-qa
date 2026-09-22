"""Read-only comparisons of saved runs with identical questions and evidence."""

def compare(left, right):
    if left.get('id') == right.get('id'):
        raise ValueError('请选择两次不同的实验')
    for run in (left, right):
        if run.get('status') != 'ready' or not run['config'].get('generate'):
            raise ValueError('两份结果都必须是已完成的生成实验')
        if not run.get('rows'):
            raise ValueError('实验没有可比较的逐题结果')
    lconfig = {k: v for k, v in left['config'].items() if k != 'expected_provider'}
    rconfig = {k: v for k, v in right['config'].items() if k != 'expected_provider'}
    if lconfig != rconfig:
        raise ValueError('两次实验题集或实验设置不一致')
    shared = ['dataset_sha256', 'elements_sha256', 'documents', 'code_sha256', 'packages',
              'prompt_sha256', 'retrieval_version', 'generation_top_k', 'max_images',
              'context_char_budget', 'temperature', 'device']
    for key in shared:
        if key not in left['provenance'] or left['provenance'][key] != right['provenance'].get(key):
            raise ValueError(f'实验来源信息不一致：{key}')
    for key in ['embedding', 'reranker', 'clip']:
        if left['provenance']['models'][key] != right['provenance']['models'][key]:
            raise ValueError(f'检索模型不一致：{key}')
    pairs = []
    indexed = [{(r['group'], r['question_id']): r for r in run['rows']} for run in (left, right)]
    if set(indexed[0]) != set(indexed[1]) or any(len(idx) != len(run['rows']) for idx, run in zip(indexed, (left, right), strict=True)):
        raise ValueError('实验题目缺失或重复')
    for key, a in indexed[0].items():
        b = indexed[1][key]
        if a['retrieved'] != b['retrieved'] or a['answer']['context'] != b['answer']['context']:
            raise ValueError(f'{key} 的实际检索或生成证据不一致')
        if a['answer']['image_count'] != b['answer']['image_count']:
            raise ValueError(f'{key} 的图片数不一致')
        item = {'question_id': key[1], 'group': key[0], 'category': a['category'],
                'question': a['question'], 'reference_answer': a.get('reference_answer'),
                'image_count': a['answer']['image_count'], 'same_evidence': True}
        for label, row in [('left', a), ('right', b)]:
            answer = row['answer']
            item.update({label + '_model': answer['model'], label + '_status': answer['status'],
                         label + '_answer': answer['text'], label + '_seconds': row['seconds'],
                         label + '_generation_seconds': answer['timings'].get('generation_seconds', 0),
                         label + '_generation_attempts': len([k for k in answer['usage'] if k.startswith('generation_')]),
                         label + '_tokens': sum(v.get('total_tokens', 0) for v in answer['usage'].values()),
                         label + '_citation_labels': ','.join(answer['citations'])})
        pairs.append(item)
    return {'left_id': left['id'], 'right_id': right['id'], 'rows': pairs,
            'checks': shared + ['retrieval_models', 'retrieved_ids', 'actual_context', 'image_count'],
            'notes': ['同题诊断抽样，不代表完整独立测试集；不自动计算语义准确率。',
                      '总耗时含检索和首次加载；对照时同时报告生成阶段耗时。',
                      'tokens 采用各服务自己的计数，不能据此直接推算人民币费用。']}
