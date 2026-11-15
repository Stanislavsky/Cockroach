
import cv2
import time

from datetime import datetime

from config.camera import CAMERA_URL

from config.detection import (
    MIN_AREA,
    CAMERA_MOVE_THRESHOLD,
    BACKGROUND_HISTORY,
    BACKGROUND_THRESHOLD,
    DETECT_SHADOWS,
    WARMUP_FRAMES,
    NOISE_KERNEL_SIZE,
    MERGE_KERNEL_SIZE,
    MOTION_CONFIRM_FRAMES
)

from config.recording import (
    RECORDING_ENABLED,
    RECORDINGS_DIR,
    SEGMENT_SECONDS,
    NO_MOTION_TIMEOUT,
    VIDEO_FORMAT
)


def main():

    # ==========================
    # ПОДКЛЮЧЕНИЕ К КАМЕРЕ
    # ==========================

    cap = cv2.VideoCapture(CAMERA_URL)

    if not cap.isOpened():
        print("Не удалось подключиться к камере")
        return

    print("Камера подключена!")

    # ==========================
    # ПОДГОТОВКА ЗАПИСИ
    # ==========================

    if RECORDING_ENABLED:
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        print("Запись включена")
    else:
        print("Режим тестирования: запись отключена")

    # ==========================
    # НАСТРОЙКА ДЕТЕКТОРА
    # ==========================

    detector = cv2.createBackgroundSubtractorMOG2(
        history=BACKGROUND_HISTORY,
        varThreshold=BACKGROUND_THRESHOLD,
        detectShadows=DETECT_SHADOWS
    )

    noise_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        NOISE_KERNEL_SIZE
    )

    merge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        MERGE_KERNEL_SIZE
    )

    # ==========================
    # ОКНА ОТОБРАЖЕНИЯ
    # ==========================

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 900, 600)

    cv2.namedWindow("Motion mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Motion mask", 600, 400)

    # ==========================
    # ПЕРЕМЕННЫЕ ЗАПИСИ
    # ==========================

    writer = None

    recording_start = None
    last_motion_time = None

    recording_enabled = RECORDING_ENABLED

    camera_moving = False

    warmup_frames = WARMUP_FRAMES

    # Счётчик подтверждения движения.
    motion_counter = 0

    # Получаем FPS камеры.
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or fps > 60:
        fps = 15.0

    try:

        # ==========================
        # ГЛАВНЫЙ ЦИКЛ ПРОГРАММЫ
        # ==========================

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Не удалось получить изображение")
                break

            now = time.monotonic()

            # Сбрасываем состояние обнаружения на каждом кадре.
            motion_detected = False

            # Сглаживаем изображение.
            blurred = cv2.GaussianBlur(
                frame,
                (5, 5),
                0
            )

            # ==========================
            # ОБУЧЕНИЕ ДЕТЕКТОРА
            # ==========================

            if warmup_frames > 0:

                mask = detector.apply(
                    blurred,
                    learningRate=0.1
                )

                warmup_frames -= 1

                cv2.putText(
                    frame,
                    f"BACKGROUND LEARNING: {warmup_frames}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.imshow("Camera", frame)
                cv2.imshow("Motion mask", mask)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                continue

            # ==========================
            # ПОЛУЧЕНИЕ МАСКИ ДВИЖЕНИЯ
            # ==========================

            learning_rate = 0.15 if camera_moving else -1

            mask = detector.apply(
                blurred,
                learningRate=learning_rate
            )

            # Удаляем мелкий шум.
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                noise_kernel
            )

            # ==========================
            # ПРОВЕРКА ДВИЖЕНИЯ КАМЕРЫ
            # ==========================

            motion_ratio = (
                cv2.countNonZero(mask) / mask.size
            )

            if motion_ratio > CAMERA_MOVE_THRESHOLD:

                camera_moving = True

                # Сбрасываем подтверждение движения.
                motion_counter = 0
                motion_detected = False

                cv2.putText(
                    frame,
                    "CAMERA MOVING",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                mask_to_show = mask

            else:

                camera_moving = False

                # ==========================
                # ОБРАБОТКА ОБЛАСТЕЙ ДВИЖЕНИЯ
                # ==========================

                # Объединяем близко расположенные области.
                merged_mask = cv2.dilate(
                    mask,
                    merge_kernel,
                    iterations=1
                )

                # Ищем контуры объектов.
                contours, _ = cv2.findContours(
                    merged_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                # Оставляем только достаточно большие контуры.
                valid_contours = [
                    contour
                    for contour in contours
                    if cv2.contourArea(contour) >= MIN_AREA
                ]

                # ==========================
                # ПОДТВЕРЖДЕНИЕ ДВИЖЕНИЯ
                # ==========================

                if valid_contours:

                    motion_counter += 1

                else:

                    motion_counter = 0

                # Подтверждаем движение после нескольких кадров.
                motion_detected = (
                    motion_counter >= MOTION_CONFIRM_FRAMES
                )

                # ==========================
                # ОТОБРАЖЕНИЕ ОБЪЕКТОВ
                # ==========================

                if valid_contours:

                    all_points = cv2.vconcat(valid_contours)

                    x, y, w, h = cv2.boundingRect(all_points)

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                mask_to_show = merged_mask

            # ==========================
            # ВЫВОД ИНФОРМАЦИИ О ДВИЖЕНИИ
            # ==========================

            cv2.putText(
                frame,
                f"Motion: {motion_counter}/{MOTION_CONFIRM_FRAMES}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            # ==========================
            # УПРАВЛЕНИЕ ВИДЕОЗАПИСЬЮ
            # ==========================

            if motion_detected:

                # Запоминаем время последнего движения.
                last_motion_time = now

                # Проверяем, разрешена ли запись.
                if recording_enabled:

                    # Если видео ещё не записывается,
                    # создаём новый видеофайл.
                    if writer is None:

                        height, width = frame.shape[:2]

                        filename = (
                            datetime.now().strftime(
                                "%Y-%m-%d_%H-%M-%S"
                            ) + ".mp4"
                        )

                        filepath = RECORDINGS_DIR / filename

                        fourcc = cv2.VideoWriter_fourcc(
                            *VIDEO_FORMAT
                        )

                        writer = cv2.VideoWriter(
                            str(filepath),
                            fourcc,
                            fps,
                            (width, height)
                        )

                        if not writer.isOpened():

                            print("Ошибка создания видеофайла")

                            writer.release()
                            writer = None

                        else:

                            recording_start = now

                            print(f"Началась запись: {filepath}")

            # ==========================
            # КОНТРОЛЬ ВРЕМЕНИ ЗАПИСИ
            # ==========================

            if writer is not None:

                # Сохраняем текущий кадр.
                writer.write(frame)

                elapsed = now - recording_start

                if last_motion_time is not None:
                    silence = now - last_motion_time
                else:
                    silence = 0

                # Проверяем максимальную длину фрагмента.
                if elapsed >= SEGMENT_SECONDS:

                    writer.release()
                    writer = None

                    recording_start = None

                    print("Пятиминутный фрагмент сохранён")

                # Проверяем время отсутствия движения.
                elif silence >= NO_MOTION_TIMEOUT:

                    writer.release()
                    writer = None

                    recording_start = None
                    last_motion_time = None

                    print(
                        "Движение прекратилось. "
                        "Запись завершена."
                    )

            # ==========================
            # ОТОБРАЖЕНИЕ СТАТУСА
            # ==========================

            if writer is not None:

                status = "RECORDING"
                color = (0, 0, 255)

            elif recording_enabled:

                status = "ARMED - WAITING MOTION"
                color = (0, 255, 255)

            else:

                status = "RECORDING OFF"
                color = (150, 150, 150)

            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.imshow("Camera", frame)
            cv2.imshow("Motion mask", mask_to_show)

            # ==========================
            # УПРАВЛЕНИЕ КЛАВИШАМИ
            # ==========================

            key = cv2.waitKey(1) & 0xFF

            # Выход из программы.
            if key == ord("q"):

                break

            # Включение / выключение записи.
            elif key in (ord("w"), ord("W")):

                recording_enabled = not recording_enabled

                if recording_enabled:

                    RECORDINGS_DIR.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    print("Запись ВКЛЮЧЕНА. Ожидание движения...")

                else:

                    # Закрываем текущий видеофайл.
                    if writer is not None:

                        writer.release()
                        writer = None

                        recording_start = None
                        last_motion_time = None

                        print("Текущий видеофрагмент сохранён")

                    print("Запись ВЫКЛЮЧЕНА")

    finally:

        # ==========================
        # ЗАВЕРШЕНИЕ ПРОГРАММЫ
        # ==========================

        if writer is not None:
            writer.release()

        cap.release()

        cv2.destroyAllWindows()

        print("Программа завершена")


if __name__ == "__main__":
    main()
