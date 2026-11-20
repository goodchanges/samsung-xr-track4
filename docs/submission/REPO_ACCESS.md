# Repository access for judges

The original repository is private. The collaborator query returned only its owner account. The existing draft contains VIT.zip (411,863,918 bytes). A draft is not a public release and a judge with ordinary read access may not be able to see it. Confirm from the intended viewer's account before submitting any link.

The final form expressly asks for a public GitHub link. Use the collaborator route only if Samsung confirms an exception.

## Checks before sharing

- The current archive has no raw videos, annotation directories, kit data, work files, or caches. The existing repository history contains nine historical annotation JSONs under docs/recognition/annotations and earlier evaluation records. Review rights across the entire history before making this repository public. A clean public snapshot with only cleared files is preferable when retaining restricted history.
- Review GitHub secret scanning and Git history before opening access. The local targeted scan checks common GitHub/OpenAI/Google token formats and private-key headers. It cannot prove that all secrets or personal data are absent. Keep member contact details in the form.
- CLIP's 605 MB checkpoint uses Git LFS. The [pinned Hugging Face model](https://huggingface.co/openai/clip-vit-base-patch32) identifies the OpenAI source but does not list a separate weight license on its model card. [OpenAI's CLIP distribution](https://github.com/openai/CLIP/blob/main/LICENSE) carries an MIT notice. Preserve upstream notices and confirm rights for redistributed Hugging Face checkpoint and tokenizer files if uncertain. The README provides a hash-checked source-download path.
- The default ZIP excludes TSM, the fitted adapter and joint decoder. The [EPIC-KITCHENS TSM distribution](https://github.com/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM) is attributed to University of Bristol and its repository uses CC BY-NC 4.0. Retain attribution and confirm noncommercial scope and the weight terms before sharing those files. [The EPIC annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/master/license.txt) have separate CC BY-NC terms.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp#license) was used locally for acquisition. Its source uses the Unlicense, while bundled builds can contain other licensed components. It is outside the runtime and ZIP. Downloading a video does not grant redistribution rights. Clear the footage shown in any public demo.
- Runtime dependencies are installed separately. Review [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE), [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt), [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE), [psutil](https://github.com/giampaolo/psutil/blob/master/LICENSE), and [imageio-ffmpeg and its bundled FFmpeg](https://github.com/imageio/imageio-ffmpeg#license) if repackaging dependency files or binaries. Keep their notices with the files you redistribute.

## Option A · DO NOT RUN · Make the existing repository public

Only after clearing the existing history and approving disclosure:

```sh
gh repo edit goodchanges/samsung-xr-track4 --visibility public --accept-visibility-change-consequences
gh repo view goodchanges/samsung-xr-track4 --json visibility,url
```

The repository and its published releases become publicly available. The draft and its VIT.zip asset remain unpublished. A separate deliberate release publication or other public placement is needed for the package to be accessible.

## Option B · DO NOT RUN · Invite confirmed organizers

Obtain the exact GitHub usernames from Samsung before inviting, and confirm that private access satisfies the form. Check the repository account's available permission controls before sending invitations.

```sh
gh api --method PUT repos/goodchanges/samsung-xr-track4/collaborators/ORGANIZER_GITHUB_USERNAME -f permission=pull
gh api repos/goodchanges/samsung-xr-track4/invitations --jq 'map(.invitee.login)'
```

An organizer must accept the invitation. Accepted collaborators can access the repository and its published private release assets subject to their role. Verify the draft asset from the judge's account, or provide the package via an agreed delivery method. Do not grant write access merely to expose a draft.
