
# task_config.py

class TaskConfig:
    """Base configuration for all tasks"""
    def __init__(
        self, 
        path, 
        name, 
        split, 
        fewshot_split, 
        answer, 
        question_name, 
        completion, 
        language,
        # NEW FLAGS:
        has_dev=False,             # True if there's a dedicated dev split
        test_with_labels=True,     # True if the 'test' split has gold labels
        split_ratio=0.8,           # For 80:20 splitting of train set
        **kwargs
    ):
        self.path = path
        self.name = name
        self.split = split
        self.fewshot_split = fewshot_split
        self.answer = answer
        self.question_name = question_name
        self.completion = completion
        self.language = language

        # New fields:
        self.has_dev = has_dev
        self.test_with_labels = test_with_labels
        self.split_ratio = split_ratio

        # We might store new splits in memory if needed
        self.new_train_data = None
        self.new_val_data = None

        # Store any additional fields
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def get_instruction(self, instruction_type, subtask_name=None):
        """Get instruction for a given type. Can be overridden by subclasses."""
        instruction_attr = f"instruction_{instruction_type}"
        if hasattr(self, instruction_attr):
            return getattr(self, instruction_attr)
        return ""


class MultiSubtaskConfig(TaskConfig):
    """Configuration for tasks with multiple subtasks (like MMLU, KMMLU)"""
    def __init__(
        self, 
        path, 
        split, 
        fewshot_split, 
        answer, 
        question_name, 
        completion, 
        language, 
        instruction_template, 
        subtask_mapping=None,
        has_dev=True,            # MMLU, KMMLU do have a dedicated dev set
        test_with_labels=True,   # MMLU, KMMLU test sets have labels
        split_ratio=0.8,
        **kwargs
    ):
        super().__init__(
            path=path,
            name=None,  # will be set on subtask
            split=split,
            fewshot_split=fewshot_split,
            answer=answer,
            question_name=question_name,
            completion=completion,
            language=language,
            has_dev=has_dev,
            test_with_labels=test_with_labels,
            split_ratio=split_ratio,
            **kwargs
        )
        self.instruction_template = instruction_template
        self.subtask_mapping = subtask_mapping or {}
        self.is_multitask = True
    
    def get_instruction(self, instruction_type, subtask_name=None):
        """Generate instruction based on subtask."""
        if subtask_name and instruction_type in self.instruction_template:
            template = self.instruction_template[instruction_type]
            # Use subtask mapping if available (e.g., for Korean names)
            display_name = self.subtask_mapping.get(subtask_name, subtask_name)
            return template.format(subject=display_name)
        return ""
    
    def create_subtask_config(self, subtask_name):
        """Create a config instance for a specific subtask."""
        config = TaskConfig(
            path=self.path,
            name=subtask_name,
            split=self.split,
            fewshot_split=self.fewshot_split,
            answer=self.answer,
            question_name=self.question_name,
            completion=self.completion,
            language=self.language,
            # Reuse parent's new fields
            has_dev=self.has_dev,
            test_with_labels=self.test_with_labels,
            split_ratio=self.split_ratio
        )
        # Store reference to parent for instruction generation
        config.parent_task = self
        config.subtask_name = subtask_name
        # Override get_instruction to use parent's template
        config.get_instruction = lambda inst_type: self.get_instruction(inst_type, subtask_name)
        return config


MULTITASK_CONFIGS = {
    "mmlu": MultiSubtaskConfig(
        path="cais/mmlu",
        split="test",
        fewshot_split="dev",
        answer="answer",
        question_name="Question",
        completion="Answer",
        language="EN",
        instruction_template={
            "single": "The following are multiple choice questions (with answers) about {subject}. Provide your answer with one of A, B, C, or D.\n\n",
            "log": "Answer the following question about {subject}.\n\n",
            "mixture": "Answer the following question about {subject}. Your answer should include both the letter and the full choice.\n\n"
        },
        # MMLU: definitely has dev set, test set has labels
        has_dev=True,
        test_with_labels=True,
    ),
    "kmmlu": MultiSubtaskConfig(
        path="HAERAE-HUB/KMMLU",
        split="test",
        fewshot_split="dev",
        answer="answer",
        question_name="문제",
        completion="답",
        language="KO",
        instruction_template={
            "single": "다음은 {subject} 분야에 관한 사지선다 문제이다. A, B, C, D 중 하나로 답하시오.\n\n",
            "mixture": "다음은 {subject} 분야에 관한 사지선다 문제이다. A, B, C, D 중 하나를 선지의 내용과 함께 답하시오.\n\n",
            "log": "다음은 {subject} 분야에 관한 문제이다. 문제에 대하여 알맞게 답하시오.\n\n"
        },
        subtask_mapping={
            'Accounting': '회계',
            'Agricultural-Sciences': '농업과학',
            'Aviation-Engineering-and-Maintenance': '항공공학 및 정비',
            'Biology': '생물학',
            'Chemical-Engineering': '화학공학',
            'Chemistry': '화학',
            'Civil-Engineering': '토목공학',
            'Computer-Science': '컴퓨터과학',
            'Construction': '건설',
            'Criminal-Law': '형법',
            'Ecology': '생태학',
            'Economics': '경제학',
            'Education': '교육학',
            'Electrical-Engineering': '전기공학',
            'Electronics-Engineering': '전자공학',
            'Energy-Management': '에너지관리',
            'Environmental-Science': '환경과학',
            'Fashion': '패션',
            'Food-Processing': '식품가공',
            'Gas-Technology-and-Engineering': '가스기술 및 공학',
            'Geomatics': '지리정보학',
            'Health': '보건',
            'Industrial-Engineer': '산업공학',
            'Information-Technology': '정보기술',
            'Interior-Architecture-and-Design': '실내건축 및 디자인',
            'Law': '법학',
            'Machine-Design-and-Manufacturing': '기계설계 및 제조',
            'Management': '경영학',
            'Maritime-Engineering': '해양공학',
            'Marketing': '마케팅',
            'Materials-Engineering': '재료공학',
            'Mechanical-Engineering': '기계공학',
            'Nondestructive-Testing': '비파괴검사',
            'Patent': '특허',
            'Political-Science-and-Sociology': '정치학 및 사회학',
            'Psychology': '심리학',
            'Public-Safety': '공공안전',
            'Railway-and-Automotive-Engineering': '철도 및 자동차공학',
            'Real-Estate': '부동산학',
            'Refrigerating-Machinery': '냉동기계',
            'Social-Welfare': '사회복지학',
            'Taxation': '세무학',
            'Telecommunications-and-Wireless-Technology': '통신 및 무선기술',
            'Korean-History': '한국사',
            'Math': '수학'
        },
        has_dev=True,
        test_with_labels=True,
    )
}

# Define single task configurations
SINGLE_TASK_CONFIGS = {
    "piqa": TaskConfig(
        path="ybisk/piqa",
        name="plain_text",
        split="test", # no label for test set
        fewshot_split="train", 
        answer=None,
        instruction_single="Select the more appropriate completion based on the context. Provide your answer with either A or B.\n\n",
        instruction_log="Provide a natural completion based on the following context.\n\n",
        instruction_mixture="Select the more appropriate completion based on the context. Your answer should include both the letter and the full choice.\n\n",
        question_name="Context",
        completion="Completion",
        language="EN",
        has_dev=False,          
        test_with_labels=False,
        split_ratio=0.8
    ),
    "hellaswag": TaskConfig(
        path="Rowan/hellaswag",
        name="default",
        split="validation", # no label for test set
        fewshot_split="train",
        answer="label",
        instruction_single="Select the most appropriate completion based on the context. Provide your answer with one of A, B, C, or D.\n\n",
        instruction_log="Provide a natural completion based on the following context.\n\n",
        instruction_mixture="Select the most appropriate completion based on the context. Your answer should include both the letter and the full choice.\n\n",
        question_name="Context",
        completion="Completion",
        language="EN",
        has_dev=False,          
        test_with_labels=False,
        split_ratio=0.8
    ),
    "winogrande": TaskConfig(
        path="allenai/winogrande",
        name="winogrande_xl",
        split="validation", # no label for test set
        fewshot_split="train",
        answer="answer",
        instruction_single='Choose the option that better completes the blank ("_") in the sentence below. Provide your answer with either A or B.\n\n',
        instruction_log='Complete the blank("_") in the following sentence.\n\n',
        instruction_mixture='Choose the option that better completes the blank ("_") in the sentence below. Your answer should include both the letter and the full choice.\n\n',
        instruction_special='Provide a natural and coherent completion to the following sentence.\n\n',
        question_name="Sentence",
        completion="Answer",
        language="EN",
        has_dev=False,          
        test_with_labels=False,
        split_ratio=0.8
    ),
    "openbookqa": TaskConfig(
        path="allenai/openbookqa",
        name="main",
        split="test",
        fewshot_split="train",
        answer="answerKey",
        instruction_single="Read the following question and choose the best answer. Provide your answer with one of A, B, C, D.\n\n",
        instruction_log="Answer the following question.\n\n",
        instruction_mixture="Read the following question and choose the best answer. Your answer should include both the letter and the full choice.\n\n",
        question_name="Question",
        completion="Answer",
        language="EN",
        has_dev=False,          
        test_with_labels=True
    ),
    "arc_easy": TaskConfig(
        path="allenai/ai2_arc",
        name="ARC-Easy",
        split="test",
        fewshot_split="train",
        answer="answerKey",
        instruction_single="Read the following science question and choose the best answer. Provide your answer with one of A, B, C, D, or E (if present).\n\n",
        instruction_log="Answer the following science question.\n\n",
        instruction_mixture="Answer the following science question. Your answer should include both the letter and the full choice.\n\n",
        question_name="Question",
        completion="Answer",
        language="EN",
        has_dev=False,          
        test_with_labels=True
    ),
    "arc_challenge": TaskConfig(
        path="allenai/ai2_arc",
        name="ARC-Challenge",
        split="test",
        fewshot_split="train",
        answer="answerKey",
        instruction_single="Read the following science question and choose the best answer. Provide your answer with one of A, B, C, D, or E (if present).\n\n",
        instruction_log="Answer the following science question.\n\n",
        instruction_mixture="Answer the following science question. Your answer should include both the letter and the full choice.\n\n",
        question_name="Question",
        completion="Answer",
        language="EN",
        has_dev=False,          
        test_with_labels=True
    ),
    "commonsense_qa": TaskConfig(
        path="tau/commonsense_qa",
        name="default",
        split="test",  # no label for test set
        fewshot_split="train",
        answer="answerKey",
        instruction_single="Read the following commonsense reasoning question and choose the best answer. Provide your answer with one of A, B, C, D, or E.\n\n",
        instruction_log="Answer the following commonsense reasoning question.\n\n",
        instruction_mixture="Answer the following commonsense reasoning question. Your answer should include both the letter and the full choice.\n\n",
        question_name="Question",
        completion="Answer",
        language="EN",
        has_dev=False,          
        test_with_labels=False,
        split_ratio=0.8
    ),
    "boolq": TaskConfig(
        path="google/boolq",
        name="default",
        split="validation", # no label for test set
        fewshot_split="train",
        answer="answer",
        instruction_single="Read the following passage and answer the yes/no question. Provide your answer with either A or B.\n\n",
        instruction_log="Read the following passage and answer the question with either Yes or No.\n\n",
        instruction_mixture="Read the following passage and answer the yes/no question. Your answer should include both the letter and the full choice.\n\n",
        question_name="",
        completion="Answer",
        language="EN",
        has_dev=False,          
        test_with_labels=False,
        split_ratio=0.8
    )
}

def get_task_config(task_name):
    """Get configuration for a task. Handles both single tasks and subtasks."""
    if task_name in SINGLE_TASK_CONFIGS:
        return SINGLE_TASK_CONFIGS[task_name]
    if task_name in MMLU_subject:
        return MULTITASK_CONFIGS["mmlu"].create_subtask_config(task_name)
    elif task_name in KMMLU_subject:
        return MULTITASK_CONFIGS["kmmlu"].create_subtask_config(task_name)
    if task_name in MULTITASK_CONFIGS:
        return MULTITASK_CONFIGS[task_name]
    raise ValueError(f"Unknown task: {task_name}")

# Backward compatibility - create information dict
information = {}
for name, config in SINGLE_TASK_CONFIGS.items():
    information[name] = config.__dict__.copy()

# Add MMLU and KMMLU to information dict for backward compatibility
information["mmlu"] = MULTITASK_CONFIGS["mmlu"].__dict__.copy()
information["kmmlu"] = MULTITASK_CONFIGS["kmmlu"].__dict__.copy()


# Subject lists
KMMLU_subject = ['Accounting', 'Agricultural-Sciences', 'Aviation-Engineering-and-Maintenance', 'Biology', 'Chemical-Engineering', 'Chemistry', 'Civil-Engineering', 'Computer-Science', 'Construction', 'Criminal-Law', 'Ecology', 'Economics', 'Education', 'Electrical-Engineering', 'Electronics-Engineering', 'Energy-Management', 'Environmental-Science', 'Fashion', 'Food-Processing', 'Gas-Technology-and-Engineering', 'Geomatics', 'Health', 'Industrial-Engineer', 'Information-Technology', 'Interior-Architecture-and-Design', 'Law', 'Machine-Design-and-Manufacturing', 'Management', 'Maritime-Engineering', 'Marketing', 'Materials-Engineering', 'Mechanical-Engineering', 'Nondestructive-Testing', 'Patent', 'Political-Science-and-Sociology', 'Psychology', 'Public-Safety', 'Railway-and-Automotive-Engineering', 'Real-Estate', 'Refrigerating-Machinery', 'Social-Welfare', 'Taxation', 'Telecommunications-and-Wireless-Technology', 'Korean-History', 'Math']

MMLU_subject = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']


# Korean language mappings
KMMLU_dict = {
    'Accounting': '회계',
    'Agricultural-Sciences': '농업과학',
    'Aviation-Engineering-and-Maintenance': '항공공학 및 정비',
    'Biology': '생물학',
    'Chemical-Engineering': '화학공학',
    'Chemistry': '화학',
    'Civil-Engineering': '토목공학',
    'Computer-Science': '컴퓨터과학',
    'Construction': '건설',
    'Criminal-Law': '형법',
    'Ecology': '생태학',
    'Economics': '경제학',
    'Education': '교육학',
    'Electrical-Engineering': '전기공학',
    'Electronics-Engineering': '전자공학',
    'Energy-Management': '에너지관리',
    'Environmental-Science': '환경과학',
    'Fashion': '패션',
    'Food-Processing': '식품가공',
    'Gas-Technology-and-Engineering': '가스기술 및 공학',
    'Geomatics': '지리정보학',
    'Health': '보건',
    'Industrial-Engineer': '산업공학',
    'Information-Technology': '정보기술',
    'Interior-Architecture-and-Design': '실내건축 및 디자인',
    'Law': '법학',
    'Machine-Design-and-Manufacturing': '기계설계 및 제조',
    'Management': '경영학',
    'Maritime-Engineering': '해양공학',
    'Marketing': '마케팅',
    'Materials-Engineering': '재료공학',
    'Mechanical-Engineering': '기계공학',
    'Nondestructive-Testing': '비파괴검사',
    'Patent': '특허',
    'Political-Science-and-Sociology': '정치학 및 사회학',
    'Psychology': '심리학',
    'Public-Safety': '공공안전',
    'Railway-and-Automotive-Engineering': '철도 및 자동차공학',
    'Real-Estate': '부동산학',
    'Refrigerating-Machinery': '냉동기계',
    'Social-Welfare': '사회복지학',
    'Taxation': '세무학',
    'Telecommunications-and-Wireless-Technology': '통신 및 무선기술',
    'Korean-History': '한국사',
    'Math': '수학'
}


# Default model list
DEFAULT_MODELS = [
    "meta-llama/Llama-3.1-8B", 
    "meta-llama/Llama-3.1-8B-Instruct", 
    "/data/s1/jyp/hf-KorV2", 
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "dnotitia/Llama-DNA-1.0-8B-Instruct",
    "mistralai/Ministral-8B-Instruct-2410",
    "microsoft/Phi-3.5-mini-instruct",
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-7b",

    "google/gemma-7b-it",

    "mistralai/Ministral-8B-Instruct-2410"
]

# Task categories
ALL_TASKS = [
    #'mmlu',
    'openbookqa', 
    "boolq",
    'hellaswag', 
    #'ko_hellaswag', 
    'winogrande', 
    #'ko_winogrande', 
    #'snu_lambada',
    'arc_easy', 
    'arc_challenge', 
    #'ko_arc_easy', 
    #'ko_arc_challenge',
    #'kmmlu',
    "commonsense_qa",
]

# Test types
TEST_TYPES = ["single", "log", "mixture"]

# Tasks that support special test type
SPECIAL_TASKS = ["winogrande", "ko_winogrande", "snu_lambada"]

# Default test configurations for specific mode based on EXAONE-7.8B
# Format: {task_name: [test_type, instruction_flag]}
DEFAULT_TEST_CONFIG = {
    "hellaswag": ["log", False],
    "ko_hellaswag": ["log", False],
    "winogrande": ["special", True],
    "ko_winogrande": ["special", True],
    "snu_lambada": ["special", True],
    "openbookqa": ["single", True],
    "arc_easy": ["single", True],
    "arc_challenge": ["single", True],
    "ko_arc_easy": ["single", True],
    "ko_arc_challenge": ["single", True],
    "mmlu": ["single", True],
    "kmmlu": ["mixture", True]
}

# Tasks that should be evaluated by subtask

SUBTASK_EVALUATIONS = ["mmlu", "kmmlu"]

# Map to identify which parent task a subtask belongs to
SUBTASK_TO_PARENT = {}
for parent, subjects in [("mmlu", MMLU_subject), ("kmmlu", KMMLU_subject)]:
    for subject in subjects:
        SUBTASK_TO_PARENT[subject] = parent

