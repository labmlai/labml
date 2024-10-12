# AppAPI Documentation

## Initialization

### `__init__(self, base_url="http://localhost:5005/api/v1")`
Initializes the `AppAPI` class with the given base URL.

## Run Endpoints

### `get_run(self, run_uuid)`
Fetches the details of a specific run.

- **Parameters:**
  - `run_uuid` (str): The unique identifier for the run.


### `update_run_data(self, run_uuid, data)`
Updates the run data.

- **Parameters:**
  - `run_uuid` (str): The unique identifier for the run.
  - `data` (dict): The data to update, containing:
    - `name` (str)
    - `comment` (str)
    - `note` (str)
    - `favourite_configs` (List[str])
    - `selected_configs` (List[str])
    - `tags` (List[str])

### `get_run_status(self, run_uuid)`
Fetches the status of a specific run.

- **Parameters:**
  - `run_uuid` (str): The unique identifier for the run.


### `get_runs(self, folder_name='default')`
- **Parameters:**
  - `folder_name` (str): default or archive

Fetches all runs.


### `archive_runs(self, run_uuids)`
Archives the specified runs.

- **Parameters:**
  - `run_uuids` (List[str]): List of run UUIDs to archive.


### `unarchive_runs(self, run_uuids)`
Unarchives the specified runs.

- **Parameters:**
  - `run_uuids` (List[str]): List of run UUIDs to unarchive.


### `delete_runs(self, run_uuids)`
Deletes the specified runs.

- **Parameters:**
  - `run_uuids` (List[str]): List of run UUIDs to delete.


## Analysis Endpoints

### `get_analysis(self, url: str, run_uuid: str, **kwargs)`
Fetches analysis data.

- **Parameters:**
  - `url` (str): The endpoint for analysis data.
  - `run_uuid` (str): The unique identifier for the run.
  - `kwargs` (dict): Additional parameters:
    - `get_all` (bool): Whether to get all indicators or only the selected indicators.
    - `current_uuid` (str): Current run UUID for comparisons (only required for comparison metrics).


### `get_preferences(self, url: str, run_uuid)`
Fetches analysis preferences.

- **Parameters:**
  - `url` (str): The endpoint for preferences.
  - `run_uuid` (str): The unique identifier for the run.


### `update_preferences(self, url: str, run_uuid, data)`
Updates analysis preferences.

- **Parameters:**
  - `url` (str): The endpoint for preferences.
  - `run_uuid` (str): The unique identifier for the run.
  - `data` (dict): The data to update, containing:
    - `series_preferences` (List[int])
    - `series_names` (List[str])
    - `chart_type` (int)
    - `step_range` (List[int])
    - `focus_smoothed` (bool)
    - `smooth_value` (float)
    - `smooth_function` (str)
    - `base_experiment` (str) (following only needed for comparison prefernces)
    - `base_series_preferences` (List[int])
    - `base_series_names` (List[str])


## Custom Metrics Endpoints

### `create_custom_metric(self, run_uuid, data)`
Creates a custom metric.

- **Parameters:**
  - `run_uuid` (str): The unique identifier for the run.
  - `data` (dict): The data for the custom metric, containing:
    - `name` (str)
    - `description` (str)

### `get_custom_metrics(self, run_uuid)`
Fetches custom metrics.

- **Parameters:**
  - `run_uuid` (str): The unique identifier for the run.


### `update_custom_metric(self, run_uuid, data)`
Updates a custom metric.

- **Parameters:**
  - `run_uuid` (str): The unique identifier for the run.
  - `data` (dict): The data to update, containing:
    - `id` (str)
    - `preferences` (dict)
    - `name` (str)
    - `description` (str)

### `delete_custom_metric(self, run_uuid, metric_id)`
Deletes a custom metric.

- **Parameters:**
  - `run_uuid` (str): The unique identifier for the run.
  - `metric_id` (str): The unique identifier for the metric.

## Logs Endpoints

### `get_logs(self, run_uuid: str, url: str, page_no: int)`
Fetches logs.

- **Parameters:**
  - `run_uuid` (str): The unique identifier for the run.
  - `url` (str): The endpoint for logs.
  - `page_no` (int): The page number to fetch:
    - `-1` -> get all pages
    - `-2` -> get last page
    - `i` -> get ith page
  