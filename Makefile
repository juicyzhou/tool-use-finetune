.PHONY: help train train-full convert stats eval eval-sample eval-cases chat deploy

help:
	@bash scripts/run.sh help

train:
	@bash scripts/run.sh train $(ARGS)

train-full:
	@bash scripts/run.sh train-full $(ARGS)

convert:
	@bash scripts/run.sh convert $(ARGS)

stats:
	@bash scripts/run.sh stats $(ARGS)

eval:
	@bash scripts/run.sh eval $(ARGS)

eval-sample:
	@bash scripts/run.sh eval-sample $(ARGS)

eval-cases:
	@bash scripts/run.sh eval-cases $(ARGS)

chat:
	@bash scripts/run.sh chat $(ARGS)

deploy:
	@bash scripts/run.sh deploy $(ARGS)
