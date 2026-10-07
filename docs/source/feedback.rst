Feedback and feature requests
=============================

Everything about rAPIdtools, from the geometry rules of the street-level
pipeline to the layout of the GUI, was shaped by people running it on real
surveys and saying what did not fit. Each form below goes to the project's
GitHub issue tracker and asks only for what is needed to act on it; a GitHub
account is the only requirement. Short notes are as welcome as long ones.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :fas:`lightbulb` Request a feature
      :link: https://github.com/RAPID-Facility/rAPIdtools/issues/new?template=feature_request.yml

      A capability that is missing or a change to an existing one. Describe
      the task and the data first; the design can follow.

   .. grid-item-card:: :fas:`bug` Report a bug
      :link: https://github.com/RAPID-Facility/rAPIdtools/issues/new?template=bug_report.yml

      Something that fails or gives a wrong result. The form asks for the
      version, the platform and the smallest script that reproduces it.

   .. grid-item-card:: :far:`comment-dots` Give feedback or ask a question
      :link: https://github.com/RAPID-Facility/rAPIdtools/issues/new?template=feedback.yml

      How the package worked on your data, a page that was hard to follow,
      a question about which component to use, or an idea that is not yet
      a proposal.

   .. grid-item-card:: :fas:`shield-halved` Report a security problem
      :link: https://github.com/RAPID-Facility/rAPIdtools/security/advisories/new

      Private by design. Vulnerabilities go through GitHub Security
      Advisories, never a public issue; see the repository's
      ``SECURITY.md``.

What happens next
-----------------

A maintainer reads every issue. Feature requests are discussed on the issue
before any code is written, so the use case and the approach are agreed on
first; if you would like to implement it yourself, the form has a box to say
so and ``CONTRIBUTING.md`` explains how a change gets in. Bugs confirmed on
real data are fixed with a test that would have caught them. Feedback on the
documentation usually lands within the next docs build; the documentation
site is regenerated from ``main`` on every push.

Other ways to reach us
----------------------

- Every page of this site has an *Open issue* entry under the repository
  button in the header, which starts an issue about that page.
- The `wiki <https://github.com/RAPID-Facility/rAPIdtools/wiki>`_ can be
  edited directly if you spot a mistake in it.
- Anything that should not be public, including suspected security problems,
  can be sent to uwrapid@uwrapid.org.

Please do not paste API keys, Mapillary tokens or Hugging Face tokens into
an issue; redact them before posting.
