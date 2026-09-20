/**
 * The stylesheet, as something a bundler will compile.
 *
 * `app.css` imports Tailwind, so it is not a file another application can
 * simply link — it has to be run through Tailwind first. Rollup will not take
 * a `.css` file as an entry point, so this one-line module is the entry, and
 * what comes out of it is `style.css`: plain CSS, already compiled, which is
 * what `@tensor-cad/ui/style.css` resolves to for anybody who installs this.
 *
 * Nothing imports this inside the repository. `main.tsx` imports the
 * stylesheet directly, because in a site build there is no packaging step to
 * need an entry for.
 */

import "./app/app.css";
