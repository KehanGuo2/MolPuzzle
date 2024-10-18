
## Example Commands (Stage 2)

<!-- ### Generate IR Questions -->

<!-- ```bash
python stage2.py --task IR --action generate_questions --input_csv ./data/mol_figures/step2.csv --output_csv ./data/mol_figures/step2/IR_questions.csv
```

### Sample Data for IR
```bash
python stage2.py --task IR --action sample_data --iterations 3
``` -->

### Generate Responses for IR Using Multiple Models
```bash
python stage2.py --task IR --action generate_responses --models instructBlip-7B instructBlip-13B llava gpt-4 claude-v1 --iterations 3
```

### Evaluate Responses for IR
```bash
python stage2.py --task IR --action evaluate --models instructBlip-7B instructBlip-13B llava gpt-4 claude-v1 --iterations 3
```