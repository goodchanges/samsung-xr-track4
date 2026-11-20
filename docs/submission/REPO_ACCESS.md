# Repository access for judges

[The GitHub repository](https://github.com/goodchanges/samsung-xr-track4) was verified public on 21 November 2026. The unpublished draft at that check held only the older VIT.zip (411,863,918 bytes). VIT-78.zip is local. Signed-out judges cannot access the draft.

The final form requires a public link. Check access while signed out, including the reviewed submission-final branch if you share it, and verify every required artifact separately.

## Checks before sharing

- The current archive has no raw videos, annotation directories, kit data, work files, or caches. The existing repository history contains nine historical annotation JSONs under docs/recognition/annotations and earlier evaluation records. Review redistribution rights for the history that is already public before distributing the submission materials.
- Review GitHub secret scanning and Git history before sharing the link. The local targeted scan checks common GitHub/OpenAI/Google token formats and private-key headers. It cannot prove that all secrets or personal data are absent. Keep member contact details in the form.
- CLIP's 605 MB checkpoint uses Git LFS. The [pinned Hugging Face model](https://huggingface.co/openai/clip-vit-base-patch32) identifies the OpenAI source but does not list a separate weight license on its model card. [OpenAI's CLIP distribution](https://github.com/openai/CLIP/blob/main/LICENSE) carries an MIT notice. Preserve upstream notices and confirm rights for redistributed Hugging Face checkpoint and tokenizer files if uncertain. The README provides a hash-checked source-download path.
- The default ZIP excludes TSM, the fitted adapter and joint decoder. The [EPIC-KITCHENS TSM distribution](https://github.com/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM) is attributed to University of Bristol and its repository uses CC BY-NC 4.0. Retain attribution and confirm noncommercial scope and the weight terms before sharing those files. [The EPIC annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/master/license.txt) have separate CC BY-NC terms.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp#license) was used locally for acquisition. Its source uses the Unlicense, while bundled builds can contain other licensed components. It is outside the runtime and ZIP. Downloading a video does not grant redistribution rights. Clear the footage shown in any public demo.
- Runtime dependencies are installed separately. Review [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE), [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt), [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE), [psutil](https://github.com/giampaolo/psutil/blob/master/LICENSE), and [imageio-ffmpeg and its bundled FFmpeg](https://github.com/imageio/imageio-ffmpeg#license) if repackaging dependency files or binaries. Keep their notices with the files you redistribute.

## Artifact access still needed

- Publish the cleared VIT-78.zip as a publicly accessible asset. The earlier attempt to upload a newer package failed with a DNS error.
- Add the completed presentation and recorded demo. The project has a local reel and script, but the reel is not the narrated submission video.
- Make the reviewed README and AI disclosure easy to find from the link you give Samsung. The submission-final branch contains the review corrections.
- If a clone leaves the 134-byte Git LFS pointer, use the tested, hash-checked Hugging Face or ZIP recovery in the [README](../../README.md).
