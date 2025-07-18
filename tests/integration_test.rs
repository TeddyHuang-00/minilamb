use minilamb::{Expr, app, evaluate, parse, reduce_once};

fn parse_(s: &str) -> Expr {
    let Ok(expr) = parse(s) else {
        panic!("Failed to parse expression: {s}");
    };
    expr
}

fn evaluate_(expr: &Expr) -> Expr {
    let Ok(result) = evaluate(expr, 1000) else {
        panic!("Failed to evaluate expression: {expr:?}");
    };
    result
}

fn parse_and_evaluate_(s: &str) -> Expr {
    evaluate_(&parse_(s))
}

#[test]
fn test_variable_capture() {
    // `(λx.x y) z` should have two free variables: `z` and `y`
    assert_eq!(parse_and_evaluate_("(λx.x y) z"), parse_("z y"));
    // `(λx y.x) y` should NOT evaluate to `λy.y`
    assert_eq!(parse_and_evaluate_("(λx y.x) y"), parse_("λx.y"));
    // `(λx y z.x) z` should NOT evaluate to `λy z.z` or `λy z.y`
    assert_eq!(parse_and_evaluate_("(λx y z.x) z"), parse_("λx y.z"));
    // `λx y.x (λx.x y)` should be parsed as `λλ.2 (λ.1 2)`
    assert_eq!(parse_("λx y.x (λx.x y)"), parse_("λλ.2 (λ.1 2)"));
}

#[test]
fn test_atomic_reduction() {
    // `(λx.x) a` -> `a` (identity returns the argument)
    assert_eq!(parse_and_evaluate_("(λx.x) a"), parse_("a"));
    // `(λx.y) a` -> `y` (constant function returns the free variable y)
    assert_eq!(parse_and_evaluate_("(λx.y) a"), parse_("y"));
    // `(λx.x x) a` -> `a a`
    assert_eq!(parse_and_evaluate_("(λx.x x) a"), parse_("a a"));
    // `(λx.y x) a` -> `y a`
    assert_eq!(parse_and_evaluate_("(λx.y x) a"), parse_("y a"));
    // `(λx.λx.x) a` -> `λx.x`
    // the second `x` shadows the first
    assert_eq!(parse_and_evaluate_("(λx.λx.x) a"), parse_("λ.1"));
    // `(λx.(λy.y) x) a` -> `(λy.y) a` -> `a`
    assert_eq!(parse_and_evaluate_("(λx.(λy.y) x) a"), parse_("a"));
}

/// i: identity
fn ski_i() -> Expr {
    parse_("λx.x")
}

/// k: constant / kestrel
fn ski_k() -> Expr {
    parse_("λx y.x")
}

/// s: distribution / starling
fn ski_s() -> Expr {
    parse_("λx y z.x z (y z)")
}

#[test]
fn test_ski_combinator() {
    // `S K K` -> `I`
    assert_eq!(evaluate_(&app!(ski_s(), ski_k(), ski_k())), ski_i());
    // `S K x y` -> `y`
    assert_eq!(
        evaluate_(&app!(ski_s(), ski_k(), parse_("x"), parse_("y"))),
        parse_("y")
    );
    // `S I I x` -> `x x`
    assert_eq!(
        evaluate_(&app!(ski_s(), ski_i(), ski_i(), parse_("x"))),
        parse_("x x")
    );
}

/// b: composition / bluebird
fn bckw_b() -> Expr {
    parse_("λx y z.x (y z)")
}

/// c: permutation / cardinal
fn bckw_c() -> Expr {
    parse_("λx y z.x z y")
}

/// w: duplication / warbler
fn bckw_w() -> Expr {
    parse_("λx y.x y y")
}

#[test]
fn test_bckw_combinator() {
    // `B f g x` -> `f (g x)`
    assert_eq!(
        evaluate_(&app!(bckw_b(), parse_("f"), parse_("g"), parse_("x"))),
        app!(parse_("f"), app!(parse_("g"), parse_("x")))
    );
    // `C f g x` -> `f x g`
    assert_eq!(
        evaluate_(&app!(bckw_c(), parse_("f"), parse_("g"), parse_("x"))),
        app!(parse_("f"), parse_("x"), parse_("g"))
    );
    // `W f x` -> `f x x`
    assert_eq!(
        evaluate_(&app!(bckw_w(), parse_("f"), parse_("x"))),
        app!(parse_("f"), parse_("x"), parse_("x"))
    );
}

fn true_() -> Expr {
    parse_("λt f.t")
}

fn false_() -> Expr {
    parse_("λt f.f")
}

fn if_() -> Expr {
    parse_("λc t f.c t f")
}

fn not_() -> Expr {
    evaluate_(&app!(
        parse_("λTRUE FALSE x.x FALSE TRUE"),
        true_(),
        false_()
    ))
}

fn and_() -> Expr {
    evaluate_(&app!(parse_("λFALSE p q.p q FALSE"), false_()))
}

fn or_() -> Expr {
    evaluate_(&app!(parse_("λTRUE p q.p TRUE q"), true_()))
}

fn xor_() -> Expr {
    evaluate_(&app!(parse_("λNOT p q.p (NOT q) q"), not_()))
}

#[test]
fn test_church_boolean() {
    // `true a (a a)` -> `a`
    assert_eq!(
        evaluate_(&app!(true_(), parse_("a"), parse_("a a"))),
        parse_("a")
    );
    // `false a (a a)` -> `a a`
    assert_eq!(
        evaluate_(&app!(false_(), parse_("a"), parse_("a a"))),
        parse_("a a")
    );
    // `if true a (a a)` -> `a`
    assert_eq!(
        evaluate_(&app!(if_(), true_(), parse_("a"), parse_("a a"))),
        parse_("a")
    );
}

#[test]
fn test_church_logic() {
    // `not true` -> `false`
    assert_eq!(evaluate_(&app!(not_(), true_())), false_());
    // `not false` -> `true`
    assert_eq!(evaluate_(&app!(not_(), false_())), true_());

    // `and true true` -> `true`
    assert_eq!(evaluate_(&app!(and_(), true_(), true_())), true_());
    // `and true false` -> `false`
    assert_eq!(evaluate_(&app!(and_(), true_(), false_())), false_());
    // `and false true` -> `false`
    assert_eq!(evaluate_(&app!(and_(), false_(), true_())), false_());
    // `and false false` -> `false`
    assert_eq!(evaluate_(&app!(and_(), false_(), false_())), false_());

    // `or true true` -> `true`
    assert_eq!(evaluate_(&app!(or_(), true_(), true_())), true_());
    // `or true false` -> `true`
    assert_eq!(evaluate_(&app!(or_(), true_(), false_())), true_());
    // `or false true` -> `true`
    assert_eq!(evaluate_(&app!(or_(), false_(), true_())), true_());
    // `or false false` -> `false`
    assert_eq!(evaluate_(&app!(or_(), false_(), false_())), false_());

    // `xor true true` -> `false`
    assert_eq!(evaluate_(&app!(xor_(), true_(), true_())), false_());
    // `xor true false` -> `true`
    assert_eq!(evaluate_(&app!(xor_(), true_(), false_())), true_());
    // `xor false true` -> `true`
    assert_eq!(evaluate_(&app!(xor_(), false_(), true_())), true_());
    // `xor false false` -> `false`
    assert_eq!(evaluate_(&app!(xor_(), false_(), false_())), false_());
}

fn zero() -> Expr {
    parse_("λs z.z")
}

fn one() -> Expr {
    parse_("λs z.s z")
}

fn two() -> Expr {
    parse_("λs z.s (s z)")
}

/// convenient function to create a numeral
fn numeral(n: usize) -> Expr {
    let mut expr = "z".to_string();
    for _ in 0..n {
        expr = format!("s ({expr})");
    }
    parse_(&format!("λs z.{expr}"))
}

/// `is_zero` Check if a numeral is zero
fn is_zero() -> Expr {
    evaluate_(&app!(
        parse_("λTRUE FALSE.λn.n (λx.FALSE) TRUE"),
        true_(),
        false_()
    ))
}

/// `succ` Add one to a numeral
fn succ() -> Expr {
    parse_("λn s z.s (n s z)")
}

/// `pred` Subtract one from a numeral
fn pred() -> Expr {
    // λn.λf.λx.n (λg.λh.h (g f)) (λu.x) (λu.u)
    parse_("λn s z.n (λg h.h (g s)) (λu.z) (λu.u)")
}

/// `add` Add two numerals
fn add() -> Expr {
    parse_("λm n s z.m s (n s z)")
}

/// `sub` Subtract two numerals
fn sub() -> Expr {
    evaluate_(&app!(parse_("λPRED.λm n.n PRED m"), pred()))
}

/// `mul` Multiply two numerals
fn mul() -> Expr {
    parse_("λm n s.m (n s)")
}

/// `pow` Exponentiation of two numerals
fn pow() -> Expr {
    parse_("λm n.n m")
}

/// `leq` Less than or equal to comparison
fn leq() -> Expr {
    evaluate_(&app!(
        parse_("λISZERO SUB.λm n.ISZERO (SUB m n)"),
        is_zero(),
        sub()
    ))
}

/// `eq` Equality comparison
fn eq() -> Expr {
    evaluate_(&app!(
        parse_("λAND LEQ.λm n.AND (LEQ m n) (LEQ n m)"),
        and_(),
        leq()
    ))
}

#[test]
fn test_church_numerals() {
    assert_eq!(evaluate_(&numeral(0)), zero());
    assert_eq!(evaluate_(&numeral(1)), one());
    assert_eq!(evaluate_(&numeral(2)), two());

    assert_eq!(evaluate_(&app!(is_zero(), numeral(0))), true_());
    assert_eq!(evaluate_(&app!(is_zero(), numeral(1))), false_());
    assert_eq!(evaluate_(&app!(is_zero(), numeral(2))), false_());

    assert_eq!(evaluate_(&app!(succ(), numeral(0))), numeral(1));
    assert_eq!(evaluate_(&app!(succ(), numeral(1))), numeral(2));
    assert_eq!(evaluate_(&app!(succ(), numeral(2))), numeral(3));

    // 0 - 1 = 0 as per Church numerals
    assert_eq!(evaluate_(&app!(pred(), numeral(0))), numeral(0));
    assert_eq!(evaluate_(&app!(pred(), numeral(1))), numeral(0));
    assert_eq!(evaluate_(&app!(pred(), numeral(2))), numeral(1));
    assert_eq!(evaluate_(&app!(pred(), numeral(3))), numeral(2));
    assert_eq!(evaluate_(&app!(pred(), numeral(4))), numeral(3));
}

#[test]
fn test_church_arithmetics_plus_minus() {
    assert_eq!(evaluate_(&app!(add(), numeral(0), numeral(0))), numeral(0));
    assert_eq!(evaluate_(&app!(add(), numeral(0), numeral(1))), numeral(1));
    assert_eq!(evaluate_(&app!(add(), numeral(1), numeral(0))), numeral(1));
    assert_eq!(evaluate_(&app!(add(), numeral(1), numeral(1))), numeral(2));
    assert_eq!(evaluate_(&app!(add(), numeral(1), numeral(2))), numeral(3));
    assert_eq!(evaluate_(&app!(add(), numeral(2), numeral(1))), numeral(3));
    assert_eq!(evaluate_(&app!(add(), numeral(2), numeral(2))), numeral(4));

    assert_eq!(evaluate_(&app!(sub(), numeral(0), numeral(0))), numeral(0));
    assert_eq!(evaluate_(&app!(sub(), numeral(0), numeral(1))), numeral(0));
    assert_eq!(evaluate_(&app!(sub(), numeral(1), numeral(0))), numeral(1));
    assert_eq!(evaluate_(&app!(sub(), numeral(1), numeral(1))), numeral(0));
    assert_eq!(evaluate_(&app!(sub(), numeral(2), numeral(1))), numeral(1));
    assert_eq!(evaluate_(&app!(sub(), numeral(2), numeral(2))), numeral(0));
    assert_eq!(evaluate_(&app!(sub(), numeral(3), numeral(2))), numeral(1));
}

#[test]
fn test_church_arithmetics_mul_pow() {
    assert_eq!(evaluate_(&app!(mul(), numeral(0), numeral(0))), numeral(0));
    assert_eq!(evaluate_(&app!(mul(), numeral(0), numeral(1))), numeral(0));
    assert_eq!(evaluate_(&app!(mul(), numeral(1), numeral(1))), numeral(1));
    assert_eq!(evaluate_(&app!(mul(), numeral(1), numeral(2))), numeral(2));
    assert_eq!(evaluate_(&app!(mul(), numeral(2), numeral(1))), numeral(2));
    assert_eq!(evaluate_(&app!(mul(), numeral(2), numeral(2))), numeral(4));
    assert_eq!(evaluate_(&app!(mul(), numeral(2), numeral(3))), numeral(6));
    assert_eq!(evaluate_(&app!(mul(), numeral(3), numeral(2))), numeral(6));

    assert_eq!(evaluate_(&app!(pow(), numeral(1), numeral(3))), numeral(1));
    assert_eq!(evaluate_(&app!(pow(), numeral(2), numeral(1))), numeral(2));
    assert_eq!(evaluate_(&app!(pow(), numeral(2), numeral(2))), numeral(4));
    assert_eq!(evaluate_(&app!(pow(), numeral(2), numeral(3))), numeral(8));
    assert_eq!(evaluate_(&app!(pow(), numeral(3), numeral(2))), numeral(9));
}

#[test]
fn test_church_equality() {
    assert_eq!(evaluate_(&app!(leq(), numeral(0), numeral(0))), true_());
    assert_eq!(evaluate_(&app!(leq(), numeral(0), numeral(1))), true_());
    assert_eq!(evaluate_(&app!(leq(), numeral(1), numeral(0))), false_());
    assert_eq!(evaluate_(&app!(leq(), numeral(1), numeral(1))), true_());
    assert_eq!(evaluate_(&app!(leq(), numeral(2), numeral(2))), true_());
    assert_eq!(evaluate_(&app!(leq(), numeral(2), numeral(3))), true_());

    assert_eq!(evaluate_(&app!(eq(), numeral(0), numeral(0))), true_());
    assert_eq!(evaluate_(&app!(eq(), numeral(0), numeral(1))), false_());
    assert_eq!(evaluate_(&app!(eq(), numeral(1), numeral(0))), false_());
    assert_eq!(evaluate_(&app!(eq(), numeral(1), numeral(1))), true_());
    assert_eq!(evaluate_(&app!(eq(), numeral(2), numeral(2))), true_());
    assert_eq!(evaluate_(&app!(eq(), numeral(2), numeral(3))), false_());
    assert_eq!(evaluate_(&app!(eq(), numeral(3), numeral(2))), false_());
}

/// `cons` Construct a list with a head and tail
fn cons() -> Expr {
    parse_("λh t f.f h t")
}

/// `nil` Empty list
fn nil() -> Expr {
    evaluate_(&app!(parse_("λTRUE.λe.TRUE"), true_()))
}

/// `is_nil` Check if a list is empty
fn is_nil() -> Expr {
    evaluate_(&app!(parse_("λFALSE.λl.l λh t.FALSE"), false_()))
}

/// `head` Get the head of a list
fn head() -> Expr {
    evaluate_(&app!(parse_("λTRUE.λl.l TRUE"), true_()))
}

/// `tail` Get the tail of a list
fn tail() -> Expr {
    evaluate_(&app!(parse_("λFALSE.λl.l FALSE"), false_()))
}

#[test]
fn test_church_lists() {
    let tuple = || app!(cons(), numeral(1), nil());
    // `head (cons 1 nil)` -> `1`
    assert_eq!(evaluate_(&app!(head(), tuple())), numeral(1));
    // `tail (cons 1 nil)` -> `nil`
    assert_eq!(evaluate_(&app!(tail(), tuple())), nil());
    // `is_nil nil` -> `true`
    assert_eq!(evaluate_(&app!(is_nil(), nil())), true_());
    // `is_nil (cons 1 nil)` -> `false`
    assert_eq!(evaluate_(&app!(is_nil(), tuple())), false_());
}

/// omega: the self-replicating combinator
fn omega() -> Expr {
    parse_("(λx.x x)(λx.x x)")
}

#[test]
fn test_self_replicating() {
    // `omega` should reduce to itself
    let Ok(Some(result)) = reduce_once(&omega()) else {
        panic!("Expected omega to reduce to itself");
    };
    assert_eq!(result, omega());

    // A very time-consuming test - omega should exceed the reduction limit
    let result = evaluate(&omega(), 100);
    assert!(result.is_err());
    let Some(err) = result.err() else {
        unreachable!()
    };
    assert_eq!(err.to_string(), "Reduction limit of 100 steps exceeded");
}

/// Y combinator: fixed-point combinator
fn y_combinator() -> Expr {
    parse_("λf.(λx.f(x x))(λx.f(x x))")
}

#[test]
fn test_y_combinator() {
    let factorial_basis = || {
        evaluate_(&app!(
            parse_("λISZERO ONE MUL PRED.λg n.(ISZERO n) ONE (MUL n(g(PRED n)))"),
            is_zero(),
            one(),
            mul(),
            pred()
        ))
    };
    let factorial = || app!(y_combinator(), factorial_basis());

    assert_eq!(evaluate_(&app!(factorial(), numeral(0))), numeral(1));
    assert_eq!(evaluate_(&app!(factorial(), numeral(1))), numeral(1));
    assert_eq!(evaluate_(&app!(factorial(), numeral(2))), numeral(2));
    assert_eq!(evaluate_(&app!(factorial(), numeral(3))), numeral(6));
}
