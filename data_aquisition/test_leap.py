import os
from Leap import LeapRecorder


def main():
    save_dir = os.path.join(os.getcwd(), 'data')
    recorder = LeapRecorder(save_dir)
    recorder.start()
    input("Press Enter to stop recording")
    recorder.stop()
    pass

if __name__ == "__main__":
    main()