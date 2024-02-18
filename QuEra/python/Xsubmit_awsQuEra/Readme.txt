The order of execution
(helper script: batchJob_aws.sh)


1) one circuit: submit to AWS  or runs local simu --> got (4)
submit_awsQuEra_job.py
  -- ProblemAtomGridSystem.py

2) retrieve_awsQuEra_job.py
3) ana_AGS.py - saves HD5 w/ shots and probs
4) pl_summaryAGS.py  - plots raw data w/o fitting

- - -  ADVANCED analysis  - - - - - 

5) aggregate_jobs.py  - concatentaes HD5's to 2D measurements

6) fit_Rabi.py - generates/verifies readErrMitt , saves calibration for it
