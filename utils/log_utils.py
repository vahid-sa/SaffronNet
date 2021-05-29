def log_history(epoch, logs, HISTORY_PATH):
    temp = {}
    for key, value in logs.items():
        temp[key] = str(value)
    with open(HISTORY_PATH, mode='a', buffering=1) as json_log:
        json_log.write(
            json.dumps({'epoch': str(epoch), **temp}) + '\n')
