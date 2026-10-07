Feedback and feature requests
=============================

Everything about rAPIdtools, from the geometry rules of the street-level
pipeline to the layout of the GUI, was shaped by people running it on real
surveys and saying what did not fit. Short notes are as welcome as long
ones. There are two ways to reach us, depending on whether you use GitHub.

Without a GitHub account
------------------------

Email is all you need. Use this route if you do not have a GitHub account or
would rather not open an issue yourself.

.. grid:: 1 1 1 1
   :gutter: 3

   .. grid-item-card:: :fas:`envelope` Write to us by email
      :link: mailto:uwrapid@uwrapid.org?subject=rAPIdtools%20feedback

      Send feedback, a question, a bug or a feature idea to
      uwrapid@uwrapid.org. Say what you were doing, what you expected and
      what happened; a screenshot or the name of the page or class helps. A
      maintainer will answer and, with your agreement, turn the message into
      a GitHub issue so others can follow it. Please leave API keys and
      access tokens out of the message.

With a GitHub account
---------------------

The three forms below open an issue on the project's GitHub page, where
other users can see it, add to it and follow the fix. Each form asks only
for what is needed to act on it. **They need a GitHub account**: clicking a
card asks you to sign in first, and GitHub offers to create a free account
on that page. If you would rather not, use the email route above.

.. grid:: 1 3 3 3
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

Security problems
-----------------

Suspected vulnerabilities are handled privately, never in a public issue.
With a GitHub account, open a `security advisory
<https://github.com/RAPID-Facility/rAPIdtools/security/advisories/new>`_;
without one, email uwrapid@uwrapid.org with "security" in the subject. The
repository's ``SECURITY.md`` describes the process.

What happens next
-----------------

A maintainer reads every email and every issue. Feature requests are
discussed before any code is written, so the use case and the approach are agreed on
first; if you would like to implement it yourself, the form has a box to say
so and ``CONTRIBUTING.md`` explains how a change gets in. Bugs confirmed on
real data are fixed with a test that would have caught them. Feedback on the
documentation usually lands within the next docs build; the documentation
site is regenerated from ``main`` on every push.

Other ways to reach us
----------------------

Both of these also need a GitHub account:

- Every page of this site has an *Open issue* entry under the repository
  button in the header, which starts an issue about that page.
- The `wiki <https://github.com/RAPID-Facility/rAPIdtools/wiki>`_ can be
  edited directly if you spot a mistake in it.

Please do not paste API keys, Mapillary tokens or Hugging Face tokens into
an email or an issue; redact them before sending.
