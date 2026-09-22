from docqa.evidence_quality import support_issues
from docqa.generation import evidence_context
from docqa.numeric_evidence import calculation_issues, table_cells, table_notes
from docqa.schemas import Element, Evidence, Source


def evidence(text, table=None):
    element = Element(id='x', document_id='d', kind='table' if table else 'text', text=text,
                      table=table, sources=[Source(page=1, bbox=(0, 0, 1, 1))])
    return Evidence(element=element, matched_text=text, document_name='d', page_label='页码',
                    rank=1, channels=[], scores={})


def test_collapsed_columns_are_aligned_and_ambiguous_rows_rejected():
    table = [['指标', '全国 城市 农村', '', ''], ['', '', '城市', '农村'],
             ['食品 交通通信', '-0.7 -2.6', '-0.5 -2.7', '-1.2 -2.5']]
    cells = table_cells(table)
    assert cells[-1] == {'row': '交通通信', 'column': '农村', 'value': '-2.5'}
    table[-1][-1] = '-1.2'
    assert table_cells(table) == []
    table[1][2] = '农村'
    assert table_cells(table) == []


def test_new_values_years_and_percent_point_computation():
    table = [['指标', '2022年', '2024年'], ['服务业', '41.25%', '46.80%']]
    notes = table_notes(table, '服务业从2022到2024提高几个百分点？')
    assert '46.80 - 41.25 = 5.55个百分点' in notes
    assert '程序计算' not in table_notes(table, '服务业2024年是多少？')
    assert '程序计算' not in table_notes(table, '制造业2022到2024提高多少？')


def test_arithmetic_uses_decimal_and_never_executes():
    assert not calculation_issues('0.3 - 0.1 = 0.2。[E1]')
    assert calculation_issues('57.7 - 54.8 = 3.9。[E1]')
    assert calculation_issues('4 / 0 = 0')
    assert not calculation_issues("__import__('os').system('echo unsafe')")


def test_wrong_table_column_detected_even_with_image_flag():
    item = evidence('交通通信表', [['指标', '全国', '城市', '农村'], ['交通通信', '-2.6', '-2.7', '-2.5']])
    _, context, _ = evidence_context([item], question='交通通信全国和农村变化？')
    context[0]['image_sent'] = True
    assert support_issues('交通通信全国为-2.6%，农村为-2.7%。[E1]', context, [item])
    assert not support_issues('交通通信全国下降2.6%，农村下降2.5%。[E1]', context, [item])


def test_citations_cannot_borrow_numbers_from_other_sentences():
    items = [evidence('甲增长1.25%。'), evidence('乙增长8.75%。')]
    items[1].element.id = 'y'
    _, context, _ = evidence_context(items)
    assert support_issues('甲增长8.75%。[E1]乙增长1.25%。[E2]', context, items)
    assert not support_issues('甲增长1.25%。[E1]乙增长8.75%。[E2]', context, items)
    # A refusal elsewhere cannot exempt an unsupported assertion.
    assert support_issues('甲增长9.87%。[E1]乙没有足够依据。', context, items)


def test_truncated_table_never_checks_unsent_rows():
    item = evidence('原文', [['指标', '城市', '农村'], ['工资', '6000', '3200']])
    _, context, _ = evidence_context([item], char_budget=0, question='工资')
    assert context[0]['text'] == ''
    assert not any('行列' in x for x in support_issues('工资农村为6000。[E1]', context, [item]))


def test_table_title_is_not_mistaken_for_the_requested_subrow():
    item = evidence('居民消费价格表', [['指标', '全国', '农村'],
                                   ['居民消费价格', '0.0', '-0.2'], ['交通通信', '-2.6', '-2.5']])
    _, context, _ = evidence_context([item], question='居民消费价格表里交通通信全国和农村变化？')
    good = '居民消费价格表里，交通通信全国的变化为-2.6%，农村的变化为-2.5%。[E1]'
    assert not support_issues(good, context, [item])
    assert support_issues(good.replace('-2.5%', '-2.7%'), context, [item])


def test_chart_difference_requires_operands_instead_of_a_guessed_result():
    question = '根据图2，2021年到2025年提高几个百分点？'
    assert support_issues('提高1.9个百分点。[E1]', [], [], question)
    assert not support_issues('无法确认图中读数。', [], [], question)
