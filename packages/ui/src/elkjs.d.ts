// elkjs ships JS only; these are the two entry points the layout module uses.
declare module "elkjs/lib/elk.bundled.js" {
  const ELK: any;
  export default ELK;
}
declare module "elkjs/lib/elk-api.js" {
  const ELK: any;
  export default ELK;
}
