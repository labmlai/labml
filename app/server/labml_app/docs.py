from labml_app.db import job

SAMPLE_SPECS_DICT = {'parameters': [], 'definitions': {}, 'response': {}}

sync = {
    "parameters": [
        {
            "name": "computer_uuid",
            "in": "query",
            "type": "string",
            "required": "true",
            "description": "0c112ffda506f10f9f793c0fb6d9de4b43595d03",
        },
        {
            "name": "runs",
            "in": "body",
            "type": "list",
            "description": "Runs to be synced with the server",
            "example": [{
                'uuid': '0c112ffda506f10f9f793c0fb6d9de4b43595d03',
                'size_tensorboard': 10.2,
                'size_checkpoints': 15.4
            }]
        },
    ],
    "responses": {
        "200": {
            "description": "Synced server side run_uuid list",
            "schema": {
                'type': 'object',
                'properties': {
                    'runs': {
                        'type': 'object',
                        'example': {
                            'active': ['0c112ffda506f10f9f793c0fb6d9de4b43595d03'],
                            'deleted': ['0c112ffda506f10f9f793c0fb6d9de4b43595d03'],
                            'unknown': ['0c112ffda506f10f9f793c0fb6d9de4b43595d03']
                        }
                    },
                }
            },
        }
    }
}

polling = {
    "parameters": [
        {
            "name": "computer_uuid",
            "in": "query",
            "type": "string",
            "required": "true",
            "description": "0c112ffda506f10f9f793c0fb6d9de4b43595d03",
        },
        {
            "name": "jobs",
            "in": "body",
            "type": "list",
            "description": "Status of the jobs initiated by UI",
            "example": [{'uuid': '0c112ffda506f10f9f793c0fb6d9de4b43595d03', 'status': job.JobStatuses.SUCCESS},
                        {'uuid': '0c112ffda506f10f9f793c0fb6d9de4b43595d03', 'status': job.JobStatuses.FAIL}]
        }
    ],
    "responses": {
        "200": {
            "description": "List of pending jobs",
            "schema": {
                'type': 'object',
                'properties': {
                    'jobs': {
                        'type': 'list',
                        'example': [
                            {
                                'uuid': '0c112ffda506f10f9f793c0fb6d9de4b43595d03',
                                'status': job.JobStatuses.INITIATED,
                                'created_time': '16234567',
                                'completed_time': None,
                                'method': job.JobMethods.START_TENSORBOARD,
                                'data': {'runs': ['0c112ffda506f10f9f793c0fb6d9de4b43595d03']}
                            }
                        ]
                    }
                }
            },
        }
    }
}

start_tensor_board = {
    "parameters": [
        {
            "name": "computer_uuid",
            "in": "path",
            "type": "string",
            "required": "true",
            "description": "0c112ffda506f10f9f793c0fb6d9de4b43595d03",
        },
        {
            "name": "runs",
            "in": "body",
            "type": "list",
            "description": "Set of runs to start TB. Note that all the runs should be from a same computer",
            "example": ['0c112ffda506f10f9f793c0fb6d9de4b43595d03']
        },
    ],
    "responses": {
        "200": {
            "description": "job details with the response",
            "schema": {
                'type': 'object',
                'example':
                    {
                        'uuid': '0c112ffda506f10f9f793c0fb6d9de4b43595d03',
                        'status': job.JobStatuses.SUCCESS,
                        'created_time': '16234567',
                        'completed_time': '16234567',
                        'method': job.JobMethods.START_TENSORBOARD
                    }
            },
        }
    }
}

clear_checkpoints = {
    "parameters": [
        {
            "name": "computer_uuid",
            "in": "path",
            "type": "string",
            "required": "true",
            "description": "0c112ffda506f10f9f793c0fb6d9de4b43595d03",
        },
        {
            "name": "runs",
            "in": "body",
            "type": "list",
            "description": "Set of runs to clear checkpoints. Note that all the runs should be from same a computer",
            "example": ['0c112ffda506f10f9f793c0fb6d9de4b43595d03']
        },
    ],
    "responses": {
        "200": {
            "description": "job details with the response",
            "schema": {
                'type': 'object',
                'example':
                    {
                        'uuid': '0c112ffda506f10f9f793c0fb6d9de4b43595d03',
                        'status': job.JobStatuses.SUCCESS,
                        'created_time': '16234567',
                        'completed_time': '16234567',
                        'method': job.JobMethods.START_TENSORBOARD
                    }
            },
        }
    }
}

get_computer = {
    "parameters": [
        {
            "name": "session_uuid",
            "in": "path",
            "type": "string",
            "required": "true",
            "description": "0c112ffda506f10f9f793c0fb6d9de4b43595d03",
        },
    ],
    "responses": {
        "200": {
            "description": "Synced server side run_uuid list",
            "schema": {
                'type': 'object',
                'example': {
                    'sessions': ['0c112ffda506f10f9f793c0fb6d9de4b43595d03',
                                 '0c112ffda506f10f9f793c0fb6d9de4b43595d03'
                                 ],
                    'session_uuid': '0c112ffda506f10f9f793c0fb6d9de4b43595d03',

                }
            },
        }
    }
}
