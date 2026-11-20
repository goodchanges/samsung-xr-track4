"""Contain inference failures without inventing observations about unreadable media."""
import sys



def main():
    try:
        from main import main as infer
        infer()
        return 0
    except (Exception, SystemExit) as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            return 0
        print(f"Track 4 failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
