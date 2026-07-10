# Contributing

## Development

Use Node.js 22 or newer and install the locked dependencies:

```sh
npm ci
```

Run the complete local validation before opening a pull request:

```sh
npm test
npm run typecheck
npm run build
npm audit
```

Tests use Node's built-in test runner and exercise the public plugin hooks. Add a regression test for behavior changes and bug fixes.

## Pull requests

Keep changes focused and document user-visible behavior in `README.md`. Do not commit generated `dist` files, cache data, credentials, or provider responses containing private metadata.

Commit messages follow Conventional Commits because semantic-release derives versions and release notes from them. Common prefixes are `fix:`, `feat:`, `docs:`, `test:`, `build:`, `ci:`, and `chore:`.

By contributing, you agree that your contribution is licensed under the MIT License.
