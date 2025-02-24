#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import qnexus as qnx
from pytket import Circuit
from pprint import pprint
from datetime import datetime
from pathlib import Path


import io

input_string = '{"id":"68505eb4-1331-4a16-9248-05da3b654322","annotations":{"name":"exec-job_2025_02_12-21-34-04","description":"","properties":{},"created":"2025-02-13 05:34:30.045605+00:00","modified":"2025-02-13 05:34:30.045605+00:00"},"job_type":"execute","last_status":"Circuit has been submitted.","last_message":"","project":{"id":"7a3de526-6bc5-46f2-b782-f8932025b770","annotations":{"name":"test-feb-12","description":"testing nexus","properties":{},"created":"2025-02-13 04:01:29.703650+00:00","modified":"2025-02-13 04:01:29.703650+00:00"},"contents_modified":"2025-02-13 05:33:26.160494+00:00","archived":false,"type":"ProjectRef"},"type":"ExecuteJobRef"}'

input_string = '{"id":"823b3736-0a45-4b73-9ce0-f9f5fa541584","annotations":{"name":"exec_2551c0","description":"","properties":{},"created":"2025-02-15 '\
                 '01:57:16.876136+00:00","modified":"2025-02-15 '\
                 '01:57:16.876136+00:00"},"job_type":"execute","last_status":"Circuit '\
                 'has been '\
                 'submitted.","last_message":"","project":{"id":"a76ed319-d3c8-4abc-ad90-5e18a6ecb3ea","annotations":{"name":"qcrank-feb-14c","description":null,"properties":{},"created":"2025-02-15 '\
                 '00:29:37.496791+00:00","modified":"2025-02-15 '\
                 '00:29:37.496791+00:00"},"contents_modified":"2025-02-15 '\
                 '01:45:06.222188+00:00","archived":false,"type":"ProjectRef"},"type":"ExecuteJobRef"}'

import json


data = json.loads(input_string)  # Works for JSON files
from qnexus.models.references import ExecuteJobRef
my_job_ref= ExecuteJobRef(**data)

print('\nstatus3:',qnx.jobs.status(my_job_ref))
