Feedback and feature requests
=============================

Much of rAPIdtools, from the geometry of the street-level pipeline to the
layout of the GUI, came from people running it on real surveys and telling
us what did not fit. A two-line note is as welcome as a long report. You can
reach us by email or, if you have a GitHub account, through an issue.

Please leave API keys, Mapillary tokens and Hugging Face tokens out of
anything you send.

By email
--------

You do not need a GitHub account for this route.

.. grid:: 1 1 1 1
   :gutter: 3

   .. grid-item-card:: :fas:`envelope` Write to us by email
      :link: mailto:uwrapid@uwrapid.org?subject=rAPIdtools%20feedback

      Send feedback, a question, a bug or a feature idea to
      uwrapid@uwrapid.org. Tell us what you were doing, what you expected
      and what happened; a screenshot, or the name of the page or class,
      helps. A maintainer will reply and, if you agree, turn your message
      into a GitHub issue so others can follow it.

On GitHub
---------

Each form below opens an issue on the project's GitHub page, where other
users can see it, add to it and follow the fix. The forms ask only for what
we need to act. GitHub asks you to sign in first, and offers to create a
free account if you do not have one.

.. grid:: 1 3 3 3
   :gutter: 3

   .. grid-item-card:: :fas:`lightbulb` Request a feature
      :link: https://github.com/RAPID-Facility/rAPIdtools/issues/new?template=feature_request.yml

      A capability that is missing, or a change to an existing one.
      Describe the task and the data first; the design can come later.

   .. grid-item-card:: :fas:`bug` Report a bug
      :link: https://github.com/RAPID-Facility/rAPIdtools/issues/new?template=bug_report.yml

      Something that fails or gives a wrong result. The form asks for the
      version, the platform and the smallest script that reproduces it.

   .. grid-item-card:: :far:`comment-dots` Give feedback or ask a question
      :link: https://github.com/RAPID-Facility/rAPIdtools/issues/new?template=feedback.yml

      How the package worked on your data, a page that was hard to follow,
      a question about which component to use, or an idea that is not yet
      a proposal.

Every page of this site also has an *Open issue* entry under the repository
button in its header, which starts an issue about that page, and you can
edit the `wiki <https://github.com/RAPID-Facility/rAPIdtools/wiki>`_
directly if you spot a mistake in it.

Security problems
-----------------

Please report suspected vulnerabilities privately, never in a public issue.
With a GitHub account, open a `security advisory
<https://github.com/RAPID-Facility/rAPIdtools/security/advisories/new>`_;
without one, email uwrapid@uwrapid.org with "security" in the subject.
``SECURITY.md`` in the repository describes the process.

What happens next
-----------------

A maintainer reads every email and every issue. Feature requests are
discussed before any code is written, so that we agree on the use case and
the approach first. If you would like to build the feature yourself, the
form has a box for that, and ``CONTRIBUTING.md`` explains how a change gets
in. Bugs confirmed on real data are fixed together with a test that would
have caught them. Fixes to the documentation usually appear with the next
docs build, since the site is rebuilt from ``main`` on every push.
