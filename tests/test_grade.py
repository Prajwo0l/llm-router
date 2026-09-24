from benchmark.grade import grade, grade_gsm8k, grade_mmlu


def test_grade_mmlu_extracts_answer_is_pattern():
    result = grade_mmlu("Let's think about this... the answer is B.", "B")
    assert result.correct is True


def test_grade_mmlu_extracts_leading_letter():
    result = grade_mmlu("C. Because photosynthesis requires sunlight.", "C")
    assert result.correct is True


def test_grade_mmlu_marks_wrong_letter_incorrect():
    result = grade_mmlu("The answer is A.", "D")
    assert result.correct is False


def test_grade_mmlu_no_letter_found_is_incorrect_not_crash():
    result = grade_mmlu("I'm not sure about this one.", "A")
    assert result.correct is False


def test_grade_gsm8k_extracts_final_number():
    result = grade_gsm8k("First we add 2 and 3 to get 5, then multiply by 4.\nThe final answer is 20", "20")
    assert result.correct is True


def test_grade_gsm8k_handles_commas_in_numbers():
    result = grade_gsm8k("The total comes out to 1,250 dollars.", "1250")
    assert result.correct is True


def test_grade_gsm8k_marks_wrong_number_incorrect():
    result = grade_gsm8k("The answer is 42.", "41")
    assert result.correct is False


def test_grade_dispatches_by_source():
    mmlu_result = grade("mmlu", "the answer is B", "B")
    gsm8k_result = grade("gsm8k", "the answer is 7", "7")
    chat_result = grade("chat", "here's a nice haiku", None)

    assert mmlu_result.correct is True
    assert gsm8k_result.correct is True
    assert chat_result.correct is None  # ungraded by design, see grade.py
