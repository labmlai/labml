from labml.internal.util import is_colab, is_kaggle
from labml.internal.lab import lab_singleton


def get_git_status():
    info = {
        'remotes': [],
        'commit': 'unknown',
        'commit_message': '',
        'is_dirty': False,
        'diff': '',
    }

    try:
        import git
    except ImportError:
        return info, 'no_gitpython'

    try:
        repo = git.Repo(lab_singleton().path)

        try:
            info['remotes'] = list(repo.remote().urls)
        except (ValueError, git.GitCommandError):
            pass
        info['commit'] = repo.head.commit.hexsha
        info['commit_message'] = repo.head.commit.message.strip()
        info['is_dirty'] = repo.is_dirty()
        info['diff'] = repo.git.diff()
    except (git.InvalidGitRepositoryError, ValueError):
        if not is_colab() and not is_kaggle():
            return info, 'no_git'
        else:
            return info, 'no_git_google_colab'

    return info, None
