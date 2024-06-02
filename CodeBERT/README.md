# SPENCER Training and Evaluation

## Cross Encoder Training

Please run the following command for cross encoder training.

```
sh train_cross_encoder.sh
```

## Dual Encoder Training

Please use the following command for dual encoder training.

```
sh train_dual_encoder.sh
```

## Model Distillation

Please run the following command for model distillation.

```
sh run_distillation.sh
```

## SPENCER Evaluation

Please run the following command to generate the prediction results for the test dataset using the cross encoder.

```
sh run_cross_encoder.sh
```

Please modify the variable `idx` in the script to ensure that prediction results are generated for the entire test dataset. After generating all the prediction results, run the following command to evaluate SPENCER.

```
sh run_SPENCER.sh
```