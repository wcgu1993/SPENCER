# SPENCER Training and Evaluation with GraphCodeBERT

The SPENCER training process includes three stages: dual-encoder training, cross-encoder training, and dual-encoder distillation. Since distillation depends on the original dual-encoder and requires the cross-encoder for performance evaluation, dual-encoder distillation begins only after both the dual-encoder and cross-encoder training are complete. Please follow the steps carefully to ensure successful model training, as any deviation may result in errors.

## Dual Encoder Training

The initial step involves training the original dual-encoder for SPENCER. To train the dual encoder, use the command provided below.

```
sh train_dual_encoder.sh
```

## Cross Encoder Training

The next step is to train the cross-encoder for SPENCER. Please use the following command to initiate the cross-encoder training process.

```
sh train_cross_encoder.sh
```

## Output the search results from Cross Encoder

The third step is to use the previously trained cross-encoder to generate prediction scores for all Code-NL pairs. In SPENCER, the distilled dual-encoder is first employed to recall code candidates, followed by the cross-encoder for re-ranking these candidates. To speed up the evaluation process during model distillation, we first have the cross-encoder predict the scores for all Code-NL pairs, then directly use these scores to re-rank the recalled code candidates. Use the following command for the cross-encoder prediction. 

```
sh run_cross_encoder.sh
```

## SPENCER Training

Once we obtain the previously trained models and results, we can officially begin the training of SPENCER. By running the following command, SPENCER will automatically be trained using the appropriate teaching assistant models and will stop when its overall performance decline exceeds the predefined maximum limit.

```
sh train_SPENCER.sh
```
In the *train_SPENCER.sh* script, two key parameters can be adjusted for SPENCER: *top_k* and *reduce_layer_num*. The *top_k* parameter specifies the number of code candidates to recall, while *reduce_layer_num* determines the number of layers to reduce during each distillation step.

```
    --top_k 5 \
    --reduce_layer_num 3 \
```

## SPENCER Evaluation

To validate the overall performance of SPENCER, execute the following command to obtain the evaluation results.

```
sh run_SPENCER.sh
```

## Pure Dual Encoder Evaluation (Optional)

To validate the performance of the original dual-encoder, run the following command to obtain the evaluation results:

```
sh run_dual_encoder.sh
```

## Pure Cross Encoder Evaluation (Optional)

To validate the performance of the pure cross-encoder, execute the following command to obtain the evaluation results:

```
python mrr.py
```
