from transformers import HfArgumentParser
from .arguments import ModelArguments, DataArguments, MyTrainingArguments
from .runner import MyRunner

parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

model_args: ModelArguments
data_args: DataArguments
training_args: MyTrainingArguments

runner = MyRunner(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args
)
runner.run()
