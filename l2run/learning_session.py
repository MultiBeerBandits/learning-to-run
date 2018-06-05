from pathlib import Path
import datetime
import json


class LearningSession:
    def __init__(self, session_path, checkpoint_dir,
                 log_dir, last_training_step, **kwargs):
        """
        Construct a LearningSession object.

        session_path Top-level directory where to save the
                     session parameters dump.

        kwargs A dictionary of the command-line arguments of the current
               learning session.
        """
        self.session_path = Path(session_path)
        if not self.session_path.is_dir():
            raise NotADirectoryError(
                "{} is not a directory".format(session_path))
        self.args = kwargs.copy()
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.last_training_step = last_training_step

    @staticmethod
    def from_last(session_path):
        """
        Construct a LearningSession object from last dump available.

        session_path Top-level directory where to load from / save to the
                     session parameters dump.
        """
        path = Path(session_path)
        if not path.is_dir():
            raise NotADirectoryError(
                "{} is not a directory".format(session_path))

        # Read each dump file name into a datetime
        dumps = [x.stem for x in path.iterdir() if x.suffix == '.json']
        name_format = '%Y%m%d-%H%M%S'

        def read_datetime(s): return datetime.datetime.strptime(
            s, name_format)

        datetimes = [read_datetime(x) for x in dumps]
        last = max(datetimes)

        # Parse json into data dict
        json_data = path / '{}.json'.format(last.strftime(name_format))
        with json_data.open(mode='r') as fin:
            data = json.load(fin)
            checkpoint_dir = data['checkpoint_dir']
            log_dir = data['log_dir']
            last_training_step = data['last_training_step']
            training_args = data['training_args']

            # Return a new instance of Learning Session
            return LearningSession(session_path,
                                   checkpoint_dir,
                                   log_dir,
                                   last_training_step,
                                   **training_args)

    @staticmethod
    def from_file(path):
        """
        Construct a LearningSession object from dump file.

        path The dump file to read LearningSession from
        """
        dump_path = Path(path)
        if not dump_path.is_file():
            raise FileNotFoundError(
                "{} is not a file".format(str(dump_path)))

        session_path = dump_path.parent
        # Parse json into data dict
        with dump_path.open(mode='r') as fin:
            data = json.load(fin)
            checkpoint_dir = data['checkpoint_dir']
            log_dir = data['log_dir']
            last_training_step = data['last_training_step']
            training_args = data['training_args']

            # Return a new instance of Learning Session
            return LearningSession(session_path,
                                   checkpoint_dir,
                                   log_dir,
                                   last_training_step,
                                   **training_args)

    def dump(self, last_training_step):
        dump_name = '{}.json'.format(self._file_name())
        save_path = self.session_path / dump_name
        self.last_training_step = last_training_step
        data = {}
        data['checkpoint_dir'] = self.checkpoint_dir
        data['log_dir'] = self.log_dir
        data['last_training_step'] = self.last_training_step
        data['training_args'] = self.args
        with save_path.open(mode='w') as fout:
            json.dump(data, fout, sort_keys=True, indent=4)

    def _file_name(self):
        now = datetime.datetime.now()
        return now.strftime('%Y%m%d-%H%M%S')
