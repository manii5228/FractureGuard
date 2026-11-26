def inference_child(channel, configuration):
    try:
        import django

        django.setup()
        from .inference import FrozenPipeline

        pipeline = FrozenPipeline(*configuration)
        channel.send(True)
        while (images := channel.recv()) is not None:
            try:
                channel.send(pipeline.predict(images))
            except Exception:
                channel.send(None)
    except (EOFError, BrokenPipeError):
        return
    except Exception:
        channel.send(None)
    finally:
        channel.close()
