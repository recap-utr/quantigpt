{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
  };
  outputs =
    inputs@{
      self,
      nixpkgs,
      flake-parts,
      systems,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import systems;
      perSystem =
        { pkgs, lib, ... }:
        let
          python = pkgs.python311;
          poetry = pkgs.poetry;
        in
        {
          devShells.default = pkgs.mkShell {
            packages = [
              python
              poetry
            ];
            POETRY_VIRTUALENVS_IN_PROJECT = true;
            shellHook = ''
              ${lib.getExe poetry} env use ${lib.getExe python}
              ${lib.getExe poetry} install --all-extras --no-root --sync
              set -a
              source .env
            '';
          };
        };
    };
}
